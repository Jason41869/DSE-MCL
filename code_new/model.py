#!/usr/bin/python3
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import utils
from utils import get_param
from decoder import ConvE
#from decoder import ConvE
#from  decoder import ComplEx
import torch.nn.functional as F
import numpy as np


class DSE-MCL(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        # get params
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']
        self.ent_emb = get_param(self.n_ent, h_dim)
        self.kg_n_layer = self.cfg.kg_layer
        self.current_h_ids = None
        self.current_t_ids = None

        # relation embedding for aggregation
        self.rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.kg_n_layer)])

        # relation embedding for prediction
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * self.kg_n_layer, h_dim)
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, h_dim)

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)
        #self.predictor = TransE(0.3)
        #self.predictor = ComplEx()
        self.bce = nn.BCELoss()
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)
        self.com_drop = nn.Dropout(self.cfg.ent_drop + 0.1)
        self.pri_drop = nn.Dropout(self.cfg.ent_drop + 0.05)
        self.act = nn.Tanh()
        self.L = nn.Linear(h_dim, h_dim)
        self.S = nn.Linear(h_dim, h_dim)
        self.alpha_c = nn.Parameter(torch.zeros(self.n_ent, 1))
        self.alpha_p = nn.Parameter(torch.zeros(self.n_ent, 1))
        self.mea_func = MLPHead(h_dim)
        self.comp_layers = nn.ModuleList([CompLayer(h_dim) for _ in range(self.kg_n_layer)])

        self.relu = nn.ReLU()
        self.W_fuse = get_param((h_dim, h_dim))
        self.a_fuse = get_param((2 * h_dim, 1))
        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.proj_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )

        # ---- MoCo 超参 ----
        self.temp = 0.15

    def forward(self, h_id, r_id, t_id, kg):
        """
        matching computation between query (h, r) and answer t.
        :param h_id: head entity id, (bs, )
        :param r_id: relation id, (bs, )
        :param kg: aggregation graph
        :return: matching score, (bs, n_ent)
        """
        # aggregate embedding
        self.current_h_ids = h_id
        self.current_t_ids = t_id
        ent_emb, rel_emb, corr, contrastive_corr = self.aggragate_emb(kg)

        head = ent_emb[h_id]
        rel = rel_emb[r_id]

        # (bs, n_ent)
        score = self.predictor(head, rel, ent_emb)

        return score, corr, contrastive_corr

    def loss(self, score, label, corr, contrastive_corr):
        alpha_l2_reg = (self.alpha_c ** 2).sum() + (self.alpha_p ** 2).sum()
        # (bs, n_ent)
        loss = self.bce(score, label) + 0.0001 * corr + 0.01 * contrastive_corr + 0.1 * alpha_l2_reg

        return loss

    def fuse_semantic_gat(self, e_original, e_sem_list):
        """
        用GAT注意力计算语义空间融合权重
        :param e_original: 当前层原始实体嵌入，shape=(n_ent, h_dim)
        :param e_sem_list: 各语义空间聚合结果列表，len=n_sem，每个元素shape=(n_ent, h_dim)
        :return: 融合后的实体嵌入，shape=(n_ent, h_dim)
        """
        # 1. 初始化融合权重列表
        fuse_weights = []

        # 2. 对每个语义空间计算GAT注意力分数
        for e_sem in e_sem_list:
            # 线性变换（GAT核心步骤）
            e_o_t = torch.matmul(e_original, self.W_fuse)  # 原始嵌入变换
            e_s_t = torch.matmul(e_sem, self.W_fuse)  # 语义空间嵌入变换

            # 拼接特征（GAT注意力输入）
            attn_input = torch.cat([e_o_t, e_s_t], dim=1)  # shape=(n_ent, 2*h_dim)

            # 计算注意力分数（LeakyReLU激活）
            score = self.LeakyReLU(torch.matmul(attn_input, self.a_fuse)).squeeze(-1)  # shape=(n_ent,)
            fuse_weights.append(score)

        # 3. Softmax归一化融合权重（每个实体的语义空间权重和为1）
        fuse_weights = torch.stack(fuse_weights, dim=1)  # shape=(n_ent, n_sem)
        fuse_weights = torch.softmax(fuse_weights, dim=1)

        # 4. 加权融合各语义空间嵌入（残差连接）
        e_fuse = e_original  # 保留原始嵌入的残差
        for i in range(len(e_sem_list)):
            e_fuse = e_fuse + fuse_weights[:, i].unsqueeze(1) * e_sem_list[i]  # 按权重加权

        return e_fuse

    def aggragate_emb(self, kg):
        ent_emb = self.ent_emb
        ent_emb_view1 = ent_emb
        ent_emb_view2 = ent_emb
        common = self.S(ent_emb)
        private = self.L(ent_emb)

        corr_loss = 0.0
        e_sem_list = []

        for i, (comp_layer, rel_emb) in enumerate(zip(self.comp_layers, self.rel_embs)):
            ent_emb_drop = self.ent_drop(ent_emb)
            rel_emb_drop = self.rel_drop(rel_emb)
            common = self.com_drop(common)
            private = self.pri_drop(private)

            comp_ent_common = comp_layer(kg, common, rel_emb_drop)
            comp_ent_private = comp_layer(kg, private, rel_emb_drop)

            ent_emb_view1 = ent_emb_view1 + self.alpha_c * comp_ent_common
            ent_emb_view2 = ent_emb_view2 + self.alpha_p * comp_ent_private

            ent_emb_sem = ent_emb_drop + comp_ent_common + comp_ent_private
            e_sem_list.append(ent_emb_sem)

            ent_emb = ent_emb_sem
            phi_c, phi_p = self.mea_func(comp_ent_common, comp_ent_private)

            corr_loss = corr_loss + compute_corr(phi_c, phi_p)

        # ===================== 对比学习 =====================

        z1 = self.proj_head(ent_emb_view1)
        z2 = self.proj_head(ent_emb_view2)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        #InfoNCE

        # 随机采样batch个实体，避免OOM & 提升效果
        # 把当前 batch 的 head 和 tail 拼起来并去重
        batch_ids = torch.unique(torch.cat([self.current_h_ids, self.current_t_ids]))
        z1 = z1[batch_ids]  # [bs, dim]
        z2 = z2[batch_ids]  # [bs, dim]

        contrastive_loss = (
                                   infonce(z1, z2.detach(), self.temp) +
                                   infonce(z2, z1.detach(), self.temp)
                           ) * 0.5
        ent_emb_fused = self.fuse_semantic_gat(ent_emb, e_sem_list)
        pred_rel_emb = torch.stack(list(self.rel_embs), dim=1).mean(dim=1)

        return ent_emb_fused, pred_rel_emb, corr_loss, contrastive_loss


class CompLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        self.comp_op = self.cfg.comp_op
        self.n_heads = self.cfg.n_heads
        self.h_dim = h_dim
        self.head_dim = self.h_dim // self.n_heads  # 480//8=60
        self.topk = self.cfg.topk
        self.W = nn.ParameterList([get_param(h_dim, self.head_dim) for _ in range(self.n_heads)])
        self.W_r = nn.ParameterList([get_param(h_dim, self.head_dim) for _ in range(self.n_heads)])
        self.a = nn.ParameterList([get_param(3 * self.head_dim, 1) for _ in range(self.n_heads)])
        assert self.comp_op in ['add', 'mul']
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.neigh_w = get_param(h_dim, h_dim)  # 480×480
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        # ========== 第一步：全图计算采样权重 ==========
        with kg.local_scope():
            kg.ndata['global_nid'] = kg.nodes().to(self.device)
            kg.ndata['emb'] = ent_emb.to(self.device)

            if 'rel_id' in kg.edata:
                rel_id = kg.edata['rel_id'].to(self.device)
            else:
                rel_id = torch.zeros(kg.number_of_edges(), dtype=torch.long).to(self.device)
            kg.edata['emb'] = rel_emb[rel_id].to(self.device)

            W0 = self.W[0]
            W_r0 = self.W_r[0]
            a0 = self.a[0]
            kg.ndata['emb_t_samp'] = torch.matmul(kg.ndata['emb'], W0)
            kg.edata['rel_t_samp'] = torch.matmul(kg.edata['emb'], W_r0)

            def cat_uv_edges_samp(edges):
                return {'u_v_cat': torch.cat([edges.src['emb_t_samp'], edges.dst['emb_t_samp']], dim=1)}

            kg.apply_edges(cat_uv_edges_samp)

            kg.edata['attn_in_samp'] = torch.cat([kg.edata['u_v_cat'], kg.edata['rel_t_samp']], dim=1)
            kg.edata['score_samp'] = self.LeakyReLU(torch.matmul(kg.edata['attn_in_samp'], a0).squeeze(-1))

            sample_kg = dgl.sampling.select_topk(kg.to('cpu'), self.topk, 'score_samp', edge_dir='in').to(self.device)

        # ========== 第二步：子图多头注意力计算 ==========
        head_outputs = []
        for head in range(self.n_heads):
            with sample_kg.local_scope():
                if 'global_nid' not in sample_kg.ndata:
                    sample_kg.ndata['global_nid'] = sample_kg.nodes().to(self.device)
                subgraph_nids = sample_kg.ndata['global_nid'].to(self.device)

                # 2. 子图特征初始化（原始emb用于线性变换，不参与组合）
                sample_kg.ndata['emb_ori'] = ent_emb[subgraph_nids].to(self.device)  # 重命名为emb_ori（480维）
                if 'rel_id' in sample_kg.edata:
                    sample_kg_rid = sample_kg.edata['rel_id'].to(self.device)
                else:
                    sample_kg_rid = torch.zeros(sample_kg.number_of_edges(), dtype=torch.long).to(self.device)
                sample_kg.edata['emb_ori'] = rel_emb[sample_kg_rid].to(self.device)  # 关系原始emb（480维）

                # 3. GAT线性变换（得到60维的emb_t/rel_t）
                W = self.W[head]
                W_r = self.W_r[head]
                a = self.a[head]
                sample_kg.ndata['emb_t'] = torch.matmul(sample_kg.ndata['emb_ori'], W)  # 60维
                sample_kg.edata['rel_t'] = torch.matmul(sample_kg.edata['emb_ori'], W_r)  # 60维

                # 4. 子图内拼接函数（用于注意力计算）
                def cat_uv_edges(edges):
                    return {'u_v_cat': torch.cat([edges.src['emb_t'], edges.dst['emb_t']], dim=1)}

                sample_kg.apply_edges(cat_uv_edges)

                # 5. 注意力计算
                sample_kg.edata['attn_in'] = torch.cat([sample_kg.edata['u_v_cat'], sample_kg.edata['rel_t']], dim=1)
                sample_kg.edata['score'] = self.LeakyReLU(torch.matmul(sample_kg.edata['attn_in'], a).squeeze(-1))
                sample_kg.edata['norm'] = dgl.ops.edge_softmax(sample_kg, sample_kg.edata['score'])

                # 6. 实体-关系组合（核心修改：用60维的emb_t/rel_t，而非480维的emb_ori）
                # 先把rel_t赋值为边的节点特征，再和emb_t组合
                sample_kg.edata['rel_t_node'] = sample_kg.edata['rel_t']  # 边的关系特征转为节点特征（60维）
                if self.comp_op == 'add':
                    # u_add_e: 源节点emb_t + 边的rel_t_node（都是60维）
                    sample_kg.apply_edges(fn.u_add_e('emb_t', 'rel_t_node', 'comp_emb'))
                else:
                    # u_mul_e: 源节点emb_t × 边的rel_t_node（都是60维）
                    sample_kg.apply_edges(fn.u_mul_e('emb_t', 'rel_t_node', 'comp_emb'))

                # 7. 加权聚合（comp_emb是60维，聚合后neigh也是60维）
                comp_emb = sample_kg.edata['comp_emb'] * sample_kg.edata['norm'].unsqueeze(-1)
                sample_kg.edata['comp_emb_non_inplace'] = comp_emb  # 新键存储
                sample_kg.update_all(fn.copy_e('comp_emb_non_inplace', 'm'), fn.sum('m', 'neigh'))

                # 8. 补全全量节点特征（此时neigh是60维，和neigh_ent_emb_head维度匹配）
                neigh_ent_emb_head = torch.zeros((self.n_ent, self.head_dim), device=self.device)
                neigh_ent_emb_head = neigh_ent_emb_head.scatter_(
                    0,  # 按第0维（实体ID）赋值
                    subgraph_nids.unsqueeze(1).expand(-1, self.head_dim),  # 扩展维度匹配
                    sample_kg.ndata['neigh']
                )
                head_outputs.append(neigh_ent_emb_head)

        # ========== 第三步：多头拼接+维度对齐 ==========
        neigh_ent_emb = torch.cat(head_outputs, dim=1)  # 14541×(8×60)=14541×480
        neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)  # 14541×480 × 480×480 → 正常

        if callable(self.bn):
            neigh_ent_emb = self.bn(neigh_ent_emb)
        neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class MLPHead(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        # 学习一个权重向量，给每个实体打分
        self.att_c = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, 1)
        )
        self.att_p = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, 1)
        )

    def forward(self, x, y):
        # x, y shape: [ent_n, h_dim]
        # 1. 计算每个实体的权重并归一化
        score_c = self.att_c(x)  # shape: [ent_n, 1]
        score_p = self.att_p(y)  # shape: [ent_n, 1]

        phi_c = score_c.squeeze(-1)  # shape: [ent_n]
        phi_p = score_p.squeeze(-1)  # shape: [ent_n]

        return phi_c, phi_p


def compute_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1 * x2)) / (sigma1 * sigma2)

    return corr


# ---- MoCo InfoNCE ----
def infonce(q, k, temp):
    # 正样本
    l_pos = (q * k).sum(dim=-1, keepdim=True)  # [N, 1]
    # 负样本
    l_neg = torch.matmul(q, k.T)  # [N, N]
    # 屏蔽自身
    l_neg = l_neg - torch.eye(l_neg.shape[0], device=l_neg.device) * 1e9

    logits = torch.cat([l_pos, l_neg], dim=1) / temp
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)




