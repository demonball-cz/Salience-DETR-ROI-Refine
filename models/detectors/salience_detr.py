from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops
from torchvision.ops import batched_nms  # 如果之后你想 NMS，这里可选

from models.bricks.denoising import GenerateCDNQueries
from models.bricks.losses import sigmoid_focal_loss
from models.detectors.base_detector import DNDETRDetector
from models.bricks.roi_refine import RoIDecoder


class SalienceCriterion(nn.Module):
    def __init__(
        self,
        limit_range: Tuple = ((-1, 64), (64, 128), (128, 256), (256, 99999)),
        noise_scale: float = 0.0, 
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.limit_range = limit_range
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, foreground_mask, targets, feature_strides, image_sizes):
        gt_boxes_list = []
        for t, (img_h, img_w) in zip(targets, image_sizes):
            boxes = t["boxes"]
            boxes = box_ops._box_cxcywh_to_xyxy(boxes)
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            gt_boxes_list.append(boxes * scale_factor)

        mask_targets = []
        for level_idx, (mask, feature_stride) in enumerate(zip(foreground_mask, feature_strides)):
            feature_shape = mask.shape[-2:]
            coord_x, coord_y = self.get_pixel_coordinate(feature_shape, feature_stride, device=mask.device)
            masks_per_level = []
            for gt_boxes in gt_boxes_list:
                mask = self.get_mask_single_level(coord_x, coord_y, gt_boxes, level_idx)
                masks_per_level.append(mask)

            masks_per_level = torch.stack(masks_per_level)
            mask_targets.append(masks_per_level)
        mask_targets = torch.cat(mask_targets, dim=1)
        foreground_mask = torch.cat([e.flatten(-2) for e in foreground_mask], -1)
        foreground_mask = foreground_mask.squeeze(1)
        num_pos = torch.sum(mask_targets > 0.5 * self.noise_scale).clamp_(min=1)
        salience_loss = (
            sigmoid_focal_loss(
                foreground_mask,
                mask_targets,
                num_pos,
                alpha=self.alpha,
                gamma=self.gamma,
            ) * foreground_mask.shape[1]
        )
        return {"loss_salience": salience_loss}

    def get_pixel_coordinate(self, feature_shape, stride, device):
        height, width = feature_shape
        coord_y, coord_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device) * stride[0],
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device) * stride[1],
            indexing="ij",
        )
        coord_y = coord_y.reshape(-1)
        coord_x = coord_x.reshape(-1)
        return coord_x, coord_y

    def get_mask_single_level(self, coord_x, coord_y, gt_boxes, level_idx):
        # gt_label: (m,) gt_boxes: (m, 4)
        # coord_x: (h*w, )
        left_border_distance = coord_x[:, None] - gt_boxes[None, :, 0]  # (h*w, m)
        top_border_distance = coord_y[:, None] - gt_boxes[None, :, 1]
        right_border_distance = gt_boxes[None, :, 2] - coord_x[:, None]
        bottom_border_distance = gt_boxes[None, :, 3] - coord_y[:, None]
        border_distances = torch.stack(
            [left_border_distance, top_border_distance, right_border_distance, bottom_border_distance],
            dim=-1,
        )  # [h*w, m, 4]

        # the foreground queries must satisfy two requirements:
        # 1. the quereis located in bounding boxes
        # 2. the distance from queries to the box center match the feature map stride
        min_border_distances = torch.min(border_distances, dim=-1)[0]  # [h*w, m]
        max_border_distances = torch.max(border_distances, dim=-1)[0]
        mask_in_gt_boxes = min_border_distances > 0
        min_limit, max_limit = self.limit_range[level_idx]
        mask_in_level = (max_border_distances > min_limit) & (max_border_distances <= max_limit)
        mask_pos = mask_in_gt_boxes & mask_in_level

        # scale-independent salience confidence
        row_factor = left_border_distance + right_border_distance
        col_factor = top_border_distance + bottom_border_distance
        delta_x = (left_border_distance - right_border_distance) / row_factor
        delta_y = (top_border_distance - bottom_border_distance) / col_factor
        confidence = torch.sqrt(delta_x**2 + delta_y**2) / 2

        confidence_per_box = 1 - confidence
        confidence_per_box[~mask_in_gt_boxes] = 0

        # process positive coordinates
        if confidence_per_box.numel() != 0:
            mask = confidence_per_box.max(-1)[0]
        else:
            mask = torch.zeros(coord_y.shape, device=confidence.device, dtype=confidence.dtype)

        # process negative coordinates
        mask_pos = mask_pos.long().sum(dim=-1) >= 1
        mask[~mask_pos] = 0

        # add noise to add randomness
        mask = (1 - self.noise_scale) * mask + self.noise_scale * torch.rand_like(mask)
        return mask


# SalienceDETR has the architecture similar to FocusDETR
class SalienceDETR(DNDETRDetector):
    def __init__(
        # model structure
        self,
        backbone: nn.Module,
        neck: nn.Module,
        position_embedding: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        focus_criterion: nn.Module,
        # model parameters
        num_classes: int = 91,
        num_queries: int = 900,
        denoising_nums: int = 100,
        # model variants
        aux_loss: bool = True,
        min_size: int = None,
        max_size: int = None,
    ):
        
        super().__init__(min_size, max_size)
        self.roi_refine_head = RoIDecoder(
            feat_channels=256,   # 一般是 transformer 的 d_model，ResNet50 默认 256
            roi_size=7,
            spatial_scale=1/8.,         # 这里先假定你用 stride=8 的特征图
        )
        self.roi_topk = 50             # 每张图选多少个框做精修，可以调
        # define model parameters
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        embed_dim = transformer.embed_dim

        # define model structures
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.denoising_generator = GenerateCDNQueries(
            num_queries=num_queries,
            num_classes=num_classes,
            label_embed_dim=embed_dim,
            denoising_nums=denoising_nums,
            label_noise_prob=0.5,
            box_noise_scale=1.0,
        )
        self.focus_criterion = focus_criterion
        self.roi_warmup_epochs = 12  # 你想要的 warmup 时长
        self.current_epoch = 0  

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # get original image sizes, used for postprocess
        original_image_sizes = self.query_original_sizes(images)
        images, targets, mask = self.preprocess(images, targets)

        # extract features
        multi_level_feats = self.backbone(images.tensors)
        multi_level_feats = self.neck(multi_level_feats)

        multi_level_masks = []
        multi_level_position_embeddings = []
        for feature in multi_level_feats:
            multi_level_masks.append(F.interpolate(mask[None], size=feature.shape[-2:]).to(torch.bool)[0])
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        if self.training:
            # collect ground truth for denoising generation
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
            noised_results = self.denoising_generator(gt_labels_list, gt_boxes_list)
            noised_label_query = noised_results[0]
            noised_box_query = noised_results[1]
            attn_mask = noised_results[2]
            denoising_groups = noised_results[3]
            max_gt_num_per_image = noised_results[4]
        else:
            noised_label_query = None
            noised_box_query = None
            attn_mask = None
            denoising_groups = None
            max_gt_num_per_image = None

        # feed into transformer
        outputs_class, outputs_coord, enc_class, enc_coord, foreground_mask, hs = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            noised_label_query,
            noised_box_query,
            attn_mask=attn_mask,
        )
        # hack implementation for distributed training
        outputs_class[0] += self.denoising_generator.label_encoder.weight[0, 0] * 0.0 
        # denoising postprocessing
        if denoising_groups is not None and max_gt_num_per_image is not None: 
            dn_metas = { 
                "denoising_groups": denoising_groups,
                "max_gt_num_per_image": max_gt_num_per_image,
            } 
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_metas) 
            # prepare for loss computation
            output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]} 
            if self.aux_loss: 
                output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord) 
            # ========= RoI 精修分支（QC-RoIFormer） =========
            if self.roi_refine_head is not None and self.training: 
                pred_logits = output["pred_logits"]      # [B, Q, num_classes+1]
                pred_boxes = output["pred_boxes"]        # [B, Q, 4], cxcywh, normalized
                B, Q, _ = pred_boxes.shape 

                # 1) 每个 query 的最高类别分数（忽略最后一个背景类）
                prob = pred_logits.softmax(-1)[..., :-1]   # [B, Q, num_classes]
                scores, _ = prob.max(-1)                   # [B, Q]

                # 2) 为每张图选出 top-K query
                topk_indices = []
                for b in range(B):
                    k = min(self.roi_topk, Q)
                    topk = torch.topk(scores[b], k=k, dim=0).indices  # [K]
                    topk_indices.append(topk)

                # 3) 网络输入尺寸（padding 后）
                H_in, W_in = images.tensors.shape[-2:]
                input_hw = (H_in, W_in)

                # 4) 选一个较高分辨率的特征图，假设 multi_level_feats[0] 是 stride=8 的最高分辨率
                high_res_feat = multi_level_feats[0]       # [B, C, Hf, Wf]

                # 5) 全局 decoder 最后一层 query 特征（用于 query-conditioned）
                global_queries = hs[-1]                    # [B, Q, C]

                # 6) 调用 QC-RoIFormer，得到 top-K 的 Δbbox
                deltas = self.roi_refine_head(
                    high_res_feat,      # [B, C, Hf, Wf]
                    pred_boxes,         # [B, Q, 4] normalized cxcywh
                    input_hw,           # (H_in, W_in)
                    topk_indices,       # List[Tensor(K,)]
                    global_queries,     # [B, Q, C]
                )   # [B*K, 4]

                # 7) 把 Δbbox 加到对应 query 的 box 上（只改 top-K，其余保持原样）
                refined_boxes = pred_boxes.clone()
                offset = 0
                for b in range(B):
                    idx = topk_indices[b]          # [K]
                    K = idx.numel()
                    delta_b = deltas[offset:offset + K]   # [K, 4]
                    offset += K
                    # 用 tanh 限制修正幅度，避免刚开始把框改飞
                    refined_boxes[b, idx] = refined_boxes[b, idx] + delta_b.tanh() * 0.1

                # 8) 构造 refine 分支；分类 logits 暂时不改，用原结果（可 detach）
                output["refine_outputs"] = {
                    "pred_logits": pred_logits.detach(),  # 不改分类，只精修 box
                    "pred_boxes": refined_boxes,
                }
            # ========= RoI 精修分支结束 ========= 
            # prepare two stage output 
            output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        if self.training:
            # compute loss
            loss_dict = self.criterion(output, targets)
            dn_losses = self.compute_dn_loss(dn_metas, targets)
            loss_dict.update(dn_losses)

            # compute focus loss
            feature_stride = [(
                images.tensors.shape[-2] / feature.shape[-2],
                images.tensors.shape[-1] / feature.shape[-1],
            ) for feature in multi_level_feats]
            focus_loss = self.focus_criterion(foreground_mask, targets, feature_stride, images.image_sizes)
            loss_dict.update(focus_loss)

            # loss reweighting
            weight_dict = self.criterion.weight_dict
            loss_dict = dict((k, loss_dict[k] * weight_dict[k]) for k in loss_dict.keys() if k in weight_dict)
            return loss_dict

        detections = self.postprocessor(output, original_image_sizes)
        return detections
