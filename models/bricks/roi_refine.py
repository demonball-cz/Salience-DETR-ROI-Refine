# models/bricks/roi_refine.py
import torch
import torch.nn as nn
from torchvision.ops import RoIAlign

class RoIDecoder(nn.Module):
    """
    Query-Conditioned RoI Transformer 解码器：
    - RoIAlign 得到局部特征 -> patch tokens
    - 将 global decoder 的 query token、roi_token 与 patch tokens 拼成一个序列
    - 用局部 Transformer 编码，最后用 roi_token 回归 Δbbox
    """
    def __init__(self,
                 feat_channels=256,
                 roi_size=7,
                 spatial_scale=1/8.,   # 对应 stride=8 特征
                 num_layers=2,
                 num_heads=8,
                 dim_feedforward=1024):
        super().__init__()
        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=spatial_scale,
            sampling_ratio=-1,
            aligned=True,
        )
        self.roi_size = roi_size
        self.feat_channels = feat_channels
        self.token_dim = feat_channels

        # 局部 Transformer（self-attention）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,   # [B, N, C]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # RoI 内 patch 的位置编码
        num_patches = roi_size * roi_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.token_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # RoI token
        self.roi_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        nn.init.trunc_normal_(self.roi_token, std=0.02)

        # 用 roi_token 回归 Δ(cx, cy, w, h)
        self.mlp_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.token_dim, 4),
        )

    def forward(self, feat, boxes, input_hw, topk_indices, global_queries):
        """
        feat: [B, C, Hf, Wf]  高分辨率特征（比如 stride=8）
        boxes: [B, Q, 4]      cxcywh, 归一化到 (H_in, W_in)
        input_hw: (H_in, W_in) 当前 batch 的网络输入尺寸（padding 后）
        topk_indices: List[Tensor]，每张图选出来的 topK query 下标
        global_queries: [B, Q, C]  全局 decoder 最后一层的 query 特征 hs[-1]
        """
        device = feat.device
        B, C, Hf, Wf = feat.shape
        H_in, W_in = input_hw

        all_rois = []
        batch_ids = []
        all_query_tokens = []

        # 1) 生成 RoI（像素坐标）及对应的 global query token
        for b in range(B):
            idx = topk_indices[b]              # [K]
            boxes_b = boxes[b, idx]            # [K, 4]
            cx, cy, bw, bh = boxes_b.unbind(-1)

            x1 = (cx - bw / 2.0) * W_in
            y1 = (cy - bh / 2.0) * H_in
            x2 = (cx + bw / 2.0) * W_in
            y2 = (cy + bh / 2.0) * H_in
            rois_b = torch.stack([x1, y1, x2, y2], dim=-1)  # [K, 4]

            all_rois.append(rois_b)
            batch_ids.append(torch.full((idx.numel(),), b, device=device, dtype=torch.float32))

            # query-conditioned：取对应 query 的 embedding 作为 query token
            qfeat_b = global_queries[b, idx]   # [K, C]
            all_query_tokens.append(qfeat_b)

        rois = torch.cat(all_rois, dim=0)                        # [B*K, 4]
        batch_ids = torch.cat(batch_ids, dim=0).unsqueeze(1)     # [B*K, 1]
        rois_input = torch.cat([batch_ids, rois], dim=1)         # [B*K, 5]

        query_tokens = torch.cat(all_query_tokens, dim=0)        # [B*K, C]

        # 2) RoIAlign 得到局部特征
        roi_feat = self.roi_align(feat, rois_input)              # [B*K, C, roi, roi]

        # 3) 构造序列：[query_token, roi_token, patch_tokens]
        Bk, C, Hroi, Wroi = roi_feat.shape
        assert Hroi == self.roi_size and Wroi == self.roi_size

        # patch tokens: [B*K, N, C]
        patch_tokens = roi_feat.view(Bk, C, -1).transpose(1, 2)  # [Bk, N, C]
        pos = self.pos_embed[:, :patch_tokens.size(1), :]        # [1, N, C]
        patch_tokens = patch_tokens + pos

        # query_token: [Bk, 1, C]
        query_token = query_tokens.unsqueeze(1)                  # [Bk, 1, C]

        # roi_token: [1,1,C] -> [Bk,1,C]
        roi_token = self.roi_token.expand(Bk, -1, -1)            # [Bk, 1, C]

        tokens = torch.cat([query_token, roi_token, patch_tokens], dim=1)  # [Bk, 2+N, C]

        # 4) 局部 Transformer 编码（self-attention）
        tokens = self.transformer(tokens)                        # [Bk, 2+N, C]

        # 5) 取第 1 个位置（roi_token）做回归
        roi_repr = tokens[:, 1, :]                               # [Bk, C]
        deltas = self.mlp_head(roi_repr)                         # [Bk, 4]

        return deltas
