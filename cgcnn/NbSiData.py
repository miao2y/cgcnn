import os
import random
from typing import List

import pandas as pd
import torch
import numpy as np
import functools
import warnings

from torch.utils.data import Dataset, DataLoader

from cgcnn.Normalizer import Normalizer
from cgcnn.data import GaussianDistance


class NbSiData(Dataset):
    adj_files: List[str] = []
    feature_folder_files: List[str] = []
    feature_folder: str = None
    adj_folder: str = None

    max_num_nbr: int = 14

    def __init__(self, root_dir, max_num_nbr=14, dmin=0, dmax=8, step=0.2,
                 random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr = max_num_nbr
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        self.adj_folder = os.path.join(self.root_dir, 'adj')
        assert os.path.exists(self.adj_folder), 'adj_folder does not exist!'

        self.feature_folder = os.path.join(self.root_dir, 'feature')
        assert os.path.exists(self.feature_folder), 'feature_folder does not exist!'
        self.adj_files = os.listdir(self.adj_folder)
        # adi_3.csv => 3
        self.adj_files.sort(key=lambda x: int(x[4:-4]))

        self.feature_files = os.listdir(self.feature_folder)
        # feature_3.txt => 3
        self.feature_files.sort(key=lambda x: int(x[8:-4]))

        random.seed(random_seed)
        # random.shuffle(self.id_prop_data)
        self.gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step)

        # return
        # print(data_id)

    def __len__(self):
        return len(self.adj_files)

    def _read_feature(self, feature_file: str):
        with open(os.path.join(self.feature_folder, feature_file), 'r') as res:
            atom_fea = res.read().split('\n')
            label = float(atom_fea[0])
            temp = [atom_fea[i + 1].split()[2:] for i in range(len(atom_fea) - 1)]
            atom_fea = Normalizer(torch.FloatTensor(np.array(temp, dtype='float')).numpy())  # norm for node feature
            return label, torch.Tensor(atom_fea.minmax_normalize())

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        feature_file = self.feature_files[idx]
        data_id = int(feature_file[8:-4])
        adj_file = os.path.join(self.adj_folder, 'adj_{}.csv'.format(data_id))
        assert os.path.exists(adj_file), 'adj_file does not exist!'
        adj = pd.read_csv(adj_file)
        adj = adj.groupby("src", group_keys=True).apply(lambda x: x)
        # print(adj)
        # print(adj[adj['src'] == 0])
        # temp = {}
        # for index, row in adj.iterrows():
        #     if not row['src'] in temp:
        #         temp[row['src']] = pd.DataFrame()
        #     temp[row['src']] = temp[row['src']] + row
        #
        # # nbr_fea = adj['weight']
        label, atom_fea = self._read_feature(feature_file)
        # print("原子数: ", atom_fea.shape[0])
        # # print("原子数2: ", len(temp.keys()))
        # max_rows = 0
        # for key in temp.keys():
        #     if max_rows < len(temp[key]):
        #         max_rows = len(temp[key])
        # print("最大边数：", max_rows)
        #
        # print("============")
        # print("atom_fea:.shape:", atom_fea.shape)

        nbr_fea = []
        nbr_fea_idx = []
        for atom_number in range(atom_fea.shape[0]):
            weights = adj[adj['src'] == atom_number]['weight'].to_list()
            neighbour_ids = adj[adj['src'] == atom_number]['tgt'].to_list()
            if len(weights) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(data_id))
                nbr_fea_idx.append(neighbour_ids + [0] * (self.max_num_nbr - len(weights)))
                nbr_fea.append(weights + [1.] * (self.max_num_nbr - len(weights)))
            else:
                nbr_fea.append(weights)
                nbr_fea_idx.append(neighbour_ids)

        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        # print(torch.Tensor(nbr_fea))
        nbr_fea = self.gdf.expand(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        # print(torch.Tensor(nbr_fea_idx))
        target = torch.Tensor([float(label)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, data_id

        # nbr_fea = []
        # for key in temp.keys():
        #     neighbours = temp[key]
        #     print(neighbours)
        #     # nbr_fea.append(neighbours['weight'])
        #     # nbr_fea.append(list)
        #
        # print("nbr_fea.shape: ", torch.Tensor(nbr_fea).shape)
        # print(nbr_fea)
        # print("atom_fea:")
        # print(atom_fea)
        # print("nbr_fea:")
        # print(nbr_fea)
        # print("总边数: ", nbr_fea.shape)
        # print("label:")
        # print(label)

        # pd.read_csv(feature_file)
        # atom_fea = res.read().split('\n')
        # adj = pd.read_csv(adj_path + 'adj_' + f.split('.')[0].split('_')[-1] + '.csv')
        # # print(features)
        #
        # nbr_fea.append(adj['weight'])
        #
        # labels.append(float(atom_fea[0]))
        #
        # # print("节点数： ", len(features) - 1)
        # # return
        # temp = [atom_fea[i + 1].split()[2:] for i in range(len(atom_fea) - 1)]
        # atom_fea = Normalizer(th.FloatTensor(np.array(temp, dtype='float')).numpy())  # norm for node feature
        # # print(features)
        # # all_feature_m.append(features_na.minmax_normalize())  # 节点特征矩阵归一化处理
        # print(th.Tensor(atom_fea.minmax_normalize()))
        # print(nbr_fea)
        # print(labels)
        # return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


if __name__ == '__main__':
    root_dir = '../data/NbSi'
    nb_si_data = NbSiData(root_dir)
    nb_si_data.__getitem__(0)
    # nb_si_data.__getitem__(1)
