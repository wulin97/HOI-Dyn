import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import pytorch3d.transforms as transforms
import numpy as np



SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class MiniTransformer(nn.Module):
    def __init__(self, n=24, dim=64, heads=4, depth=1, coupled=False):
        super().__init__()
        self.joint_encoder = nn.Sequential(
            nn.Linear(3 if n==24 else 9, dim // 2) if not coupled else nn.Linear(3 + 9, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )

        self.cond_encoder = nn.Sequential(
            nn.Linear(12+dim+4, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.pos_embed = nn.Parameter(torch.randn(1, n, dim))

        # 单层Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 注意力池化层
        self.attn_pool = nn.Linear(dim, 1)

    def forward(self, joints, cond):
        # 1. 特征增强 (注入条件信息)
        joint_feats = self.joint_encoder(joints)
        x = joint_feats + self.cond_encoder(cond).unsqueeze(1)  # [B,24,64]

        # 2. 添加位置编码
        x = x + self.pos_embed

        # 3. 通过微型Transformer
        x = self.transformer(x)  # [B,24,64]

        # 4. 注意力池化
        attn_weights = F.softmax(self.attn_pool(x).squeeze(-1), dim=1)  # [B,24]
        global_feat = (x * attn_weights.unsqueeze(-1)).sum(1)  # [B,64]

        return global_feat, attn_weights


class Dynamics(nn.Module):
    def __init__(self, output_size, feature_dim=64, heads=8, depth=4):
        super(Dynamics, self).__init__()
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=1024*3, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=feature_dim),
            )

        self.human_joints_transformer = MiniTransformer(n=24, dim=feature_dim, heads=heads, depth=depth, coupled=True)

        self.tran_rot_enc = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=feature_dim // 2),
            nn.ReLU(),
            nn.Linear(in_features=feature_dim // 2, out_features=output_size[0] + output_size[1]),
        )

    def predict(self, x, bps, max_step, method):
        pass

    def forward(self, x, dataset, max_step=1, method="theta"):
        pass


