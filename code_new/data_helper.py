import logging
import math
from collections import defaultdict
from itertools import chain
from os.path import join, exists
from typing import List, Tuple
import dgl
import numpy as np
import torch
from torch.utils.data import Dataset
import utils
import os



# ================= 全局缓存 =================
_DICT_CACHE = {}
_DATA_CACHE = {}
_FORMAT_CACHE = {}  # 缓存每个目录的列格式


def detect_column_order(paths):
    """
    智能探测函数：自动识别关系是在第二列还是第三列
    """
    if not paths: return 1, 2

    col2_vals = set()
    col3_vals = set()

    with open(paths[0], 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) == 3:
                col2_vals.add(parts[1])
                col3_vals.add(parts[2])
            if i > 300: break  # 采样前300行足够判别

    # 哪一列的不同取值数量少，哪一列就是关系（Relation）
    if len(col2_vals) < len(col3_vals):
        print("💡 【格式探测器】检测到当前数据集列顺序为标准格式: [Head, Relation, Tail]")
        return 1, 2  # rel_idx=1, tail_idx=2
    else:
        print("💡 【格式探测器】检测到当前数据集列顺序为颠倒格式: [Head, Tail, Relation] -> 已自动纠偏！")
        return 2, 1  # rel_idx=2, tail_idx=1


def get_dataset_paths(dir_path):
    """自适应探测目录下存在的数据集文件"""

    def find_file(names, exts):
        for name in names:
            for ext in exts:
                p = join(dir_path, f"{name}{ext}")
                if exists(p): return p
        return None

    train_p = find_file(['train'], ['.txt', '.triples', '.tsv'])
    valid_p = find_file(['valid', 'dev'], ['.txt', '.triples', '.tsv'])
    test_p = find_file(['test'], ['.txt', '.triples', '.tsv'])
    relation_p = find_file(['1_n'], ['.txt', '.triples', '.tsv'])
    return train_p, valid_p, test_p, relation_p


def construct_dict(dir_path):
    """
    智能构建字典：自适应文本/ID、空格/Tab、列顺序颠倒
    """
    if dir_path in _DICT_CACHE:
        return _DICT_CACHE[dir_path]

    ent2id, rel2id = dict(), dict()
    train_path, valid_path, test_path, _ = get_dataset_paths(dir_path)
    paths = [p for p in [train_path, valid_path, test_path] if p is not None]

    if not paths:
        raise FileNotFoundError(f"❌ 错误：在 {dir_path} 未找到 train/valid/test 文件！")

    # 1. 自动探测列顺序
    rel_idx, tail_idx = detect_column_order(paths)
    _FORMAT_CACHE[dir_path] = (rel_idx, tail_idx)

    # 2. 自动探测数据集是否为纯 ID 格式
    is_id_format = False
    with open(paths[0], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) == 3:
                h, r, t = parts[0], parts[rel_idx], parts[tail_idx]
                if h.isdigit() and r.isdigit() and t.isdigit():
                    is_id_format = True
                break

    # 3. 建立映射
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) != 3: continue
                h, r, t = parts[0], parts[rel_idx], parts[tail_idx]

                if is_id_format:
                    ent2id[h] = int(h)
                    ent2id[t] = int(t)
                    rel2id[r] = int(r)
                else:
                    if h not in ent2id: ent2id[h] = len(ent2id)
                    if t not in ent2id: ent2id[t] = len(ent2id)
                    if r not in rel2id: rel2id[r] = len(rel2id)

    ent2id = dict(sorted(ent2id.items(), key=lambda x: x[1]))
    rel2id = dict(sorted(rel2id.items(), key=lambda x: x[1]))
    _DICT_CACHE[dir_path] = (ent2id, rel2id)
    return ent2id, rel2id


def read_data(set_flag):
    """
    通用数据读取器：完美适配任何格式排列
    """
    assert set_flag in ['train', 'valid', 'test', ['train', 'valid'], ['train', 'valid', 'test']]
    cache_key = tuple(set_flag) if isinstance(set_flag, list) else set_flag
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    cfg = utils.get_global_config()
    dir_p = join(cfg.dataset_dir, cfg.dataset)
    ent2id, rel2id = construct_dict(dir_p)

    print(f"🔥 【智能适配器】当前数据集: {cfg.dataset} | 实体总数: {len(ent2id)}, 关系总数: {len(rel2id)}")

    # 获取该目录缓存的正确列索引
    rel_idx, tail_idx = _FORMAT_CACHE.get(dir_p, (1, 2))

    train_p, valid_p, test_p, relation_p = get_dataset_paths(dir_p)
    if set_flag == 'train':
        file = open(train_p, 'r', encoding='utf-8')
    elif set_flag == 'valid':
        file = open(valid_p, 'r', encoding='utf-8')
    elif set_flag == 'test':
        file = open(test_p, 'r', encoding='utf-8')
    elif set_flag == ['train', 'valid']:
        file = chain(open(train_p, 'r', encoding='utf-8'), open(valid_p, 'r', encoding='utf-8'))
    elif set_flag == ['train', 'valid', 'test']:
        file = chain(open(train_p, 'r', encoding='utf-8'), open(valid_p, 'r', encoding='utf-8'),
                     open(test_p, 'r', encoding='utf-8'))

    src_list, dst_list, rel_list = [], [], []
    pos_tails, pos_heads, pos_rels = defaultdict(set), defaultdict(set), defaultdict(set)

    for line in file:
        line = line.strip()
        if not line: continue
        parts = line.split()
        if len(parts) != 3: continue
        h_str, r_str, t_str = parts[0], parts[rel_idx], parts[tail_idx]

        h, r, t = ent2id[h_str], rel2id[r_str], ent2id[t_str]
        src_list.append(h)
        dst_list.append(t)
        rel_list.append(r)

        pos_tails[(h, r)].add(t)
        pos_heads[(r, t)].add(h)
        pos_rels[(h, t)].add(r)
        pos_rels[(t, h)].add(r + len(rel2id))

    output_dict = {
        'src_list': src_list, 'dst_list': dst_list, 'rel_list': rel_list,
        'pos_tails': pos_tails, 'pos_heads': pos_heads, 'pos_rels': pos_rels
    }
    _DATA_CACHE[cache_key] = output_dict
    return output_dict


def construct_kg(set_flag, directed=False):
    assert directed in [True, False]
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
    d = read_data(set_flag)
    src_list, dst_list, rel_list = [], [], []

    eid = 0
    hr2eid, rt2eid = defaultdict(list), defaultdict(list)
    for h, t, r in zip(d['src_list'], d['dst_list'], d['rel_list']):
        if directed:
            src_list.extend([h])
            dst_list.extend([t])
            rel_list.extend([r])
            hr2eid[(h, r)].extend([eid])
            rt2eid[(r, t)].extend([eid])
            eid += 1
        else:
            src_list.extend([h, t])
            dst_list.extend([t, h])
            rel_list.extend([r, r + n_rel])
            hr2eid[(h, r)].extend([eid, eid + 1])
            rt2eid[(r, t)].extend([eid, eid + 1])
            eid += 2

    src, dst, rel = torch.tensor(src_list), torch.tensor(dst_list), torch.tensor(rel_list)
    return src, dst, rel, hr2eid, rt2eid


def get_kg(src, dst, rel, device):
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
    kg = dgl.graph((src, dst), num_nodes=n_ent)
    kg.edata['rel_id'] = rel
    kg = kg.to(device)
    return kg


class TrainDataset(Dataset):
    def __init__(self, set_flag, hr2eid, rt2eid):
        assert set_flag in ['train', 'valid', 'test']
        logging.info('---Load Train Data---')
        self.cfg = utils.get_global_config()
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.d = read_data(set_flag)
        self.query = []
        self.label = []
        self.rm_edges = []

        for k, v in self.d['pos_tails'].items():
            self.query.append((k[0], k[1], -1))
            self.label.append(list(v))
            self.rm_edges.append(hr2eid[k])

        for k, v in self.d['pos_heads'].items():
            self.query.append((k[1], k[0] + self.n_rel, -1))
            self.label.append(list(v))
            self.rm_edges.append(rt2eid[k])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        h, r, t = self.query[item]
        label_indices = self.label[item]

        rm_edges = torch.tensor(self.rm_edges[item], dtype=torch.int64)
        rm_num = math.ceil(rm_edges.shape[0] * self.cfg.rm_rate)
        rm_inds = torch.randperm(rm_edges.shape[0])[:rm_num]
        rm_edges = rm_edges[rm_inds]

        return (h, r, t), label_indices, rm_edges

    def get_onehot_label(self, label):
        onehot_label = torch.zeros(self.n_ent)
        onehot_label[label] = 1
        if self.cfg.label_smooth != 0.0:
            onehot_label = (1.0 - self.cfg.label_smooth) * onehot_label + (1.0 / self.n_ent)
        return onehot_label

    def get_pos_inds(self, label):
        pos_inds = torch.zeros(self.n_ent).to(torch.bool)
        pos_inds[label] = True
        return pos_inds

    @staticmethod
    def collate_fn(data):
        src = [d[0][0] for d in data]
        rel = [d[0][1] for d in data]
        dst = [d[0][2] for d in data]
        raw_labels = [d[1] for d in data]
        rm_edges = [d[2] for d in data]

        src = torch.tensor(src, dtype=torch.int64)
        rel = torch.tensor(rel, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)

        cfg = utils.get_global_config()
        n_ent = utils.DATASET_STATISTICS[cfg.dataset]['n_ent']

        label_tensor = torch.zeros(len(data), n_ent, dtype=torch.float32)
        for i, label_list in enumerate(raw_labels):
            label_tensor[i, label_list] = 1.0

        if cfg.label_smooth != 0.0:
            label_tensor = (1.0 - cfg.label_smooth) * label_tensor + (1.0 / n_ent)

        rm_edges = torch.cat(rm_edges, dim=0)

        return (src, rel, dst), label_tensor, rm_edges


class EvalDataset(Dataset):
    def __init__(self, set_flag, mode):
        assert set_flag in ['train', 'valid', 'test']
        assert mode in ['head_batch', 'tail_batch']
        self.cfg = utils.get_global_config()
        dataset = self.cfg.dataset
        self.mode = mode
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.d = read_data(set_flag)
        self.trip = [_ for _ in zip(self.d['src_list'], self.d['rel_list'], self.d['dst_list'])]
        self.d_all = read_data(['train', 'valid', 'test'])
        self.pos_t = self.d_all['pos_tails']
        self.pos_h = self.d_all['pos_heads']

    def __len__(self):
        return len(self.trip)

    def __getitem__(self, item):
        h, r, t = self.trip[item]
        filter_bias = torch.zeros(self.n_ent, dtype=torch.float32)

        if self.mode == 'tail_batch':
            filter_bias[list(self.pos_t[(h, r)])] = -float('inf')
            filter_bias[t] = 0.
        elif self.mode == 'head_batch':
            filter_bias[list(self.pos_h[(r, t)])] = -float('inf')
            filter_bias[h] = 0.
            h, r, t = t, r + self.n_rel, h
        else:
            raise NotImplementedError

        return (h, r, t), filter_bias, self.mode

    @staticmethod
    def collate_fn(data: List[Tuple[tuple, torch.Tensor, str]]):
        h = [d[0][0] for d in data]
        r = [d[0][1] for d in data]
        t = [d[0][2] for d in data]

        filter_bias = torch.stack([d[1] for d in data], dim=0)
        mode = data[0][-1]

        h = torch.tensor(h, dtype=torch.int64)
        r = torch.tensor(r, dtype=torch.int64)
        t = torch.tensor(t, dtype=torch.int64)

        return (h, r, t), filter_bias, mode


class BiDataloader(object):
    def __init__(self, h_loader: iter, t_loader: iter):
        self.h_loader_len = len(h_loader)
        self.t_loader_len = len(t_loader)
        self.h_loader_step = 0
        self.t_loader_step = 0
        self.total_len = self.h_loader_len + self.t_loader_len
        self.h_loader = self.inf_loop(h_loader)
        self.t_loader = self.inf_loop(t_loader)
        self._step = 0

    def __next__(self):
        if self._step == self.total_len:
            assert self.h_loader_step == self.h_loader_len
            assert self.t_loader_step == self.t_loader_len
            self._step = 0
            self.h_loader_step = 0
            self.t_loader_step = 0
            raise StopIteration
        if self._step % 2 == 0:
            if self.h_loader_step < self.h_loader_len:
                data = next(self.h_loader)
                self.h_loader_step += 1
            else:
                data = next(self.t_loader)
                self.t_loader_step += 1
        else:
            if self.t_loader_step < self.t_loader_len:
                data = next(self.t_loader)
                self.t_loader_step += 1
            else:
                data = next(self.h_loader)
                self.h_loader_step += 1
        self._step += 1
        return data

    def __iter__(self):
        return self

    def __len__(self):
        return self.total_len

    @staticmethod
    def inf_loop(dataloader):
        while True:
            for data in dataloader:
                yield data