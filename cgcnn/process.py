# import dgl

import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch as th
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset
import sklearn.preprocessing as skpre
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix, diags, eye
import copy

from cgcnn.Normalizer import Normalizer

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')


def draw_graph(g):
    plt.figure(figsize=(14, 6))
    nxg = g.to_networkx()
    pos = nx.spring_layout(nxg)
    # nx.draw(, with_labels=True)
    color = ['r' for _ in range(8)] + ['y' for _ in range(6)] + ['g' for i in range(12)]
    edgewidth = list(map(lambda x: 1 / x, g.edges[:].data['weight']))

    nx.draw_networkx_edges(nxg, pos, width=edgewidth,
                           edge_color=color)  # 奇怪的是：edgewidth应该是一个列表，为什么可以直接拿来用呢？Python怎么判断哪个边用的是哪个权重值？
    nx.draw_networkx_nodes(nxg, pos)
    nx.draw_networkx_labels(nxg, pos)

    plt.show()


# ------------------------------------
# 2.重写dataset函数，定义不同的collate方法
# ------------------------------------
class My_dataset(Dataset):
    def __init__(self, graphs, labels):
        super().__init__()
        self.src = graphs
        self.trg = labels

    def __getitem__(self, index):
        # print(index)
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src)


def collate2(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    # 每个图包含三个部分：图的特征 （15，85），图的邻接矩阵表示（15，15），图的全局特征（291）
    # zip(*samples)是解压操作，解压为[(graph1, graph2, ...), (label1, label2, ...)]
    graphs, labels = map(list, zip(*samples))

    # dgl.batch 将一批图看作是具有许多互不连接的组件构成的大型图
    # return dgl.batch(graphs), th.tensor(labels, dtype=th.float32)
    return graphs, th.FloatTensor(labels)


def collate3(samples):
    '''
    获得Nb预测数据，调用dgl包装图模型
    '''
    # graphs = map(list, zip(*samples))
    gs, lgs = list(map(list, zip(*samples)))
    # dgl.batch 将一批图看作是具有许多互不连接的组件构成的大型图
    return (dgl.batch(gs), dgl.batch(lgs))


def collate4(samples):
    return samples


def get_train_val_test_data(datasets, labels, co_fn, seed, split=[0.7, 0.2, 0.1]):
    '''
    分割数据集
    '''
    co_fn = globals()[co_fn]
    assert datasets is not None and len(datasets) > 0, '数据集不能为空'
    lens = len(datasets)

    assert abs(sum(split) - 1) < 1e-5 or sum(split) == lens, 'split分割总和不为1或len(datasets)'
    # if sum(split) != lens:
    split = list(map(lambda x: int(x * lens), split))
    split[-1] = lens - sum(split[:-1])  # 确保总和为所有数据量

    n_train, n_val, n_test = split
    np.random.seed(seed)
    idxs = np.random.permutation(lens)  # 将原有索引打乱顺序

    # 计算每个数据集的索引
    idx_train = th.LongTensor(idxs[:n_train])
    idx_val = th.LongTensor(idxs[n_train:n_train + n_val])
    idx_test = th.LongTensor(idxs[n_train + n_val:])

    train_data, valid_data, test_data = [My_dataset([datasets[i] for i in j], [labels[i] for i in j]) for j in
                                         [idx_train, idx_val, idx_test]]

    train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=co_fn)
    valid_data_loader = DataLoader(valid_data, batch_size=1, shuffle=True, collate_fn=co_fn)
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=co_fn)
    print('train:{},valid:{},test:{}'.format(len(train_data_loader), len(valid_data_loader), len(test_data_loader)))
    return train_data_loader, valid_data_loader, test_data_loader


def get_pre_data(datasets, labels, co_fn):
    co_fn = globals()[co_fn]
    assert datasets is not None and len(datasets) > 0, '数据集不能为空'
    lens = len(datasets)

    # lens = 10
    idxs = np.arange(0, lens)

    train_data = My_dataset([datasets[i] for i in idxs], [labels[i] for i in idxs])

    train_data_loader = DataLoader(train_data, batch_size=1, shuffle=False, collate_fn=co_fn)
    print('train:{}'.format(len(train_data_loader)))
    return train_data_loader


# -------------
# 数据预处理
# -------------
def data_process(adj_path, feature_path, seed):
    labels = []
    nbr_fea = []
    all_feature_m = []
    weight_m = None

    for i, f in enumerate(os.listdir(feature_path)):
        if i % 50 == 0:
            print('{}/{}:{}'.format(i, len(os.listdir(feature_path)), f))
        # 针对每个晶体：
        with open(feature_path + f, 'r') as res:
            atom_fea = res.read().split('\n')
            adj = pd.read_csv(adj_path + 'adj_' + f.split('.')[0].split('_')[-1] + '.csv')
            # print(features)

            nbr_fea.append(adj['weight'])

            labels.append(float(atom_fea[0]))

            # print("节点数： ", len(features) - 1)
            # return
            temp = [atom_fea[i + 1].split()[2:] for i in range(len(atom_fea) - 1)]
            atom_fea = Normalizer(th.FloatTensor(np.array(temp, dtype='float')).numpy())  # norm for node feature
            # print(features)
            # all_feature_m.append(features_na.minmax_normalize())  # 节点特征矩阵归一化处理
            print(th.Tensor(atom_fea.minmax_normalize()))
            print(nbr_fea)
            print(labels)
            return

    # labels = np.array(labels).reshape(-1, 1)

    # draw_graph(g_mask)
    #
    # a = list(zip(all_feature_m, all_adj_m))
    # labels = labels * 10
    # # labels = normalizer(np.array(labels).reshape(-1, 1)).absmax_normalize()  # NbSi数据做归一化处理
    # # labels = np.array(labels) + 1 # Nb数据负值预测结果不好，给每个label加上一个固 定值
    # train_data_loader, valid_data_loader, test_data_loader = get_train_val_test_data(a, labels, co_fn='collate2',
    #                                                                                  seed=seed)
    # # train_data_loader = get_pre_data(a, labels, co_fn='collate2')
    #
    # return train_data_loader, valid_data_loader, test_data_loader


if __name__ == '__main__':
    data_process(adj_path="../data/NbSi/adj/", feature_path="../data/NbSi/feature/", seed=8)
