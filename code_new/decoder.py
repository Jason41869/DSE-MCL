import torch
import torch.nn as nn
import utils
import torch.nn.functional as F


class ConvE(nn.Module):
    def __init__(self, h_dim, out_channels, ker_sz):
        super().__init__()
        cfg = utils.get_global_config()
        self.cfg = cfg
        dataset = cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm1d(h_dim)

        self.conv_drop = torch.nn.Dropout(cfg.conv_drop)
        self.fc_drop = torch.nn.Dropout(cfg.fc_drop)
        self.k_h = cfg.k_h
        self.k_w = cfg.k_w
        assert self.k_h * self.k_w == h_dim
        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, stride=1, padding=0,
                                    kernel_size=ker_sz, bias=False)
        flat_sz_h = int(2 * self.k_h) - ker_sz + 1
        flat_sz_w = self.k_w - ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * out_channels
        self.fc = torch.nn.Linear(self.flat_sz, h_dim, bias=False)
        self.ent_drop = nn.Dropout(cfg.ent_drop_pred)

    def forward(self, head, rel, all_ent):
        # head (bs, h_dim), rel (bs, h_dim)
        # concatenate and reshape to 2D
        c_head = head.view(-1, 1, head.shape[-1])
        c_rel = rel.view(-1, 1, rel.shape[-1])
        c_emb = torch.cat([c_head, c_rel], 1)
        c_emb = torch.transpose(c_emb, 2, 1).reshape((-1, 1, 2 * self.k_h, self.k_w))

        x = self.bn0(c_emb)
        x = self.conv(x)  # (bs, out_channels, out_h, out_w)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)
        x = x.view(-1, self.flat_sz)  # (bs, out_channels * out_h * out_w)
        x = self.fc(x)  # (bs, h_dim)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc_drop(x)  # (bs, h_dim)
        # inference
        # all_ent: (n_ent, h_dim)
        all_ent = self.ent_drop(all_ent)
        x = torch.mm(x, all_ent.transpose(1, 0))  # (bs, n_ent)
        x = torch.sigmoid(x)
        return x

# class TransE(nn.Module):
#     def __init__(self, ent_drop_pred=0.0):
#         super().__init__()
#         self.ent_drop = nn.Dropout(ent_drop_pred)
#
#         # 🌟 核心：为了兼容 BCE/Sigmoid 引入的一维缩放和平移参数
#         self.scale = nn.Parameter(torch.Tensor([1.0]))
#         self.bias = nn.Parameter(torch.Tensor([0.0]))
#
#     def forward(self, head, rel, all_ent):
#         all_ent = self.ent_drop(all_ent)
#         hr = head + rel
#
#         # 上一步推导的高效矩阵乘法 (L2 距离平方)
#         hr_sq = torch.sum(hr ** 2, dim=-1, keepdim=True)
#         t_sq = torch.sum(all_ent ** 2, dim=-1, keepdim=True).transpose(1, 0)
#         m_prod = 2 * torch.mm(hr, all_ent.transpose(1, 0))
#
#         # 此时的 distance 是正数（距离越小越好）
#         distance = hr_sq + t_sq - m_prod
#
#         # 🌟 关键转换：通过可学习的 scale 和 bias，把“距离”线性映射成适合 Sigmoid 的输入
#         # 距离越小，得分越高
#         scores = self.bias - self.scale * distance
#
#         # 如果你的 Loss 层没自带 Sigmoid，就解开下面这行
#         scores = torch.sigmoid(scores)
#
#         return scores

