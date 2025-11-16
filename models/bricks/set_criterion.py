import copy
from typing import Dict

import torch
import torch.distributed
from torch import nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from models.bricks.losses import sigmoid_focal_loss, vari_sigmoid_focal_loss
from util.utils import get_world_size, is_dist_avail_and_initialized


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict,
        alpha: float = 0.25,
        gamma: float = 2.0,
        two_stage_binary_cls=False,
    ):
        """Create the criterion.

        :param num_classes: number of object categories, omitting the special no-object category
        :param matcher: module able to compute a matching between targets and proposals
        :param weight_dict: dict containing as key the names of the losses and as values their relative weight
        :param alpha: alpha in Focal Loss, defaults to 0.25
        :param gamma: gamma in Focal loss, defaults to 2.0
        :param two_stage_binary_cls: Whether to use two-stage binary classification loss, defaults to False
        """        
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma
        self.two_stage_binary_cls = two_stage_binary_cls

    def loss_labels(self, outputs, targets, num_boxes, indices, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_class = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.alpha,
                gamma=self.gamma,
            ) * src_logits.shape[1]
        )
        losses = {"loss_class": loss_class}
        return losses

    def loss_boxes(self, outputs, targets, num_boxes, indices, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops._box_cxcywh_to_xyxy(src_boxes),
                box_ops._box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_mir(self, outputs, targets, num_boxes):
        if "aux_outputs" not in outputs:
            return {"loss_mir": outputs["pred_boxes"].sum() * 0.0}

        matching_outputs = {k: v for k, v in outputs.items()
                            if k != "aux_outputs" and k != "enc_outputs"}
        gt_boxes, gt_labels = list(zip(*map(lambda x: (x["boxes"], x["labels"]), targets)))
        pred_logits = matching_outputs["pred_logits"]
        pred_boxes = matching_outputs["pred_boxes"]
        indices = list(map(self.matcher, pred_boxes, pred_logits, gt_boxes, gt_labels))

        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        )

        all_pred_boxes = []
        for aux_out in outputs["aux_outputs"]:
            all_pred_boxes.append(aux_out["pred_boxes"])
        all_pred_boxes.append(outputs["pred_boxes"])

        iou_per_layer = []
        for pred in all_pred_boxes:
            src_boxes = pred[idx]
            iou_mat = box_ops.box_iou(
                box_ops._box_cxcywh_to_xyxy(src_boxes),
                box_ops._box_cxcywh_to_xyxy(target_boxes),
            )
            ious = torch.diag(iou_mat)  # [num_matched]
            iou_per_layer.append(ious)

        if len(iou_per_layer) <= 1:
            return {"loss_mir": outputs["pred_boxes"].sum() * 0.0}

        # ----------✅ STEP A: 计算用于过滤的 IoU（比如最后一层） ----------
        iou_last = iou_per_layer[-1].detach()          # [num_matched]
        mask = iou_last > 0.3                          # 只保留 IoU>=0.3 的匹配

        # 如果当前 batch 一个都没达到阈值，就直接返回 0 loss
        if mask.sum() == 0:
            return {"loss_mir": outputs["pred_boxes"].sum() * 0.0}

        # ----------✅ STEP B: 只对 mask 内样本计算单调约束 ----------
        mir_loss = 0.0
        margin = 0.01  # 允许一点点波动，不至于太硬

        for l in range(len(iou_per_layer) - 1):
            iou_l = iou_per_layer[l][mask]
            iou_next = iou_per_layer[l + 1][mask]
            diff = (iou_l - iou_next - margin).clamp(min=0.0)
            mir_loss = mir_loss + diff.sum()

        # ----------✅ STEP C: 归一化 ----------
        mir_loss = mir_loss / num_boxes

        return {"loss_mir": mir_loss}



    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def calculate_loss(self, outputs, targets, num_boxes, indices=None, **kwargs):
        losses = {}
        # get matching results for each image
        if not indices:
            gt_boxes, gt_labels = list(zip(*map(lambda x: (x["boxes"], x["labels"]), targets)))
            pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
            indices = list(map(self.matcher, pred_boxes, pred_logits, gt_boxes, gt_labels))
        loss_class = self.loss_labels(outputs, targets, num_boxes, indices=indices)
        loss_boxes = self.loss_boxes(outputs, targets, num_boxes, indices=indices)
        losses.update(loss_class)
        losses.update(loss_boxes)
        return losses

    def loss_bbox_only_refine(self, refine_outputs, targets, num_boxes, final_indices, base_boxes):
        """
        只计算 refine 后 bbox/GIoU 的损失：
        - 复用最终层的 matching indices(final_indices)
        - 只在 IoU>=0.3 的 matched 上计算（门控）
        base_boxes: 最终层未 refine 的 boxes（用于计算门控 IoU）
        """
        # 门控：计算最终层（base）与 GT 的 IoU，用于 mask
        idx = self._get_src_permutation_idx(final_indices)
        base_src = base_boxes[idx]
        tgt_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, final_indices)], dim=0)

        with torch.no_grad():
            base_iou = torch.diag(
                box_ops.box_iou(
                    box_ops._box_cxcywh_to_xyxy(base_src),
                    box_ops._box_cxcywh_to_xyxy(tgt_boxes),
                )
            )
            gate = base_iou > 0.3
            if gate.sum() == 0:
                return {"loss_bbox_refine": refine_outputs["pred_boxes"].sum() * 0.0,
                        "loss_giou_refine": refine_outputs["pred_boxes"].sum() * 0.0}

        # 只取门控内的样本
        ref_src = refine_outputs["pred_boxes"][idx][gate]
        tgt_sel = tgt_boxes[gate]

        loss_bbox = F.l1_loss(ref_src, tgt_sel, reduction="none").sum() / num_boxes
        loss_giou = (1.0 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops._box_cxcywh_to_xyxy(ref_src),
                box_ops._box_cxcywh_to_xyxy(tgt_sel),
            )
        )).sum() / num_boxes

        return {"loss_bbox_refine": loss_bbox, "loss_giou_refine": loss_giou}

    def forward(self, outputs, targets):
        """This performs the loss computation

        :param outputs: dict of tensors, see the output specification of the model for the format
        :param targets: list of dicts, such that len(targets) == batch_size
        :return: a dict containing losses
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            data=[num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        matching_outputs = {k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"}
        gt_boxes, gt_labels = list(zip(*map(lambda x: (x["boxes"], x["labels"]), targets)))
        pred_logits = matching_outputs["pred_logits"]
        pred_boxes = matching_outputs["pred_boxes"]
        final_indices = list(map(self.matcher, pred_boxes, pred_logits, gt_boxes, gt_labels))

        # 1) 原损失：保持原逻辑(每层各自匹配)
        losses = {}
        losses.update(self.calculate_loss(matching_outputs, targets, num_boxes))
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                losses_aux = self.calculate_loss(aux_outputs, targets, num_boxes)
                losses.update({k + f"_{i}": v for k, v in losses_aux.items()})

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            if self.two_stage_binary_cls:
                for bt in bin_targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            losses_enc = self.calculate_loss(enc_outputs, bin_targets, num_boxes)
            losses.update({k + f"_enc": v for k, v in losses_enc.items()})

        # 2) MIR（可选）：用 final_indices；你已有 loss_mir，就不改了
        if "aux_outputs" in outputs and "loss_mir" in self.weight_dict:
            losses.update(self.loss_mir(outputs, targets, num_boxes))

        # 3) RoI refine：仅 bbox/GIoU + 复用 final_indices + IoU门控
        if "refine_outputs" in outputs and "loss_bbox_refine" in self.weight_dict:
            losses.update(self.loss_bbox_only_refine(outputs["refine_outputs"], targets, num_boxes, final_indices, pred_boxes))

        return losses


class HybridSetCriterion(SetCriterion):
    def loss_labels(self, outputs, targets, num_boxes, indices, **kwargs):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou_score = torch.diag(
            box_ops.box_iou(
                box_ops._box_cxcywh_to_xyxy(src_boxes),
                box_ops._box_cxcywh_to_xyxy(target_boxes),
            )
        ).detach()  # add detach according to RT-DETR

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        # construct onehot targets, shape: (batch_size, num_queries, num_classes)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]

        # construct iou_score, shape: (batch_size, num_queries)
        target_score = torch.zeros_like(target_classes, dtype=iou_score.dtype)
        target_score[idx] = iou_score

        loss_class = (
            vari_sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                target_score,
                num_boxes=num_boxes,
                alpha=self.alpha,
                gamma=self.gamma,
            ) * src_logits.shape[1]
        )
        losses = {"loss_class": loss_class}
        return losses
