import os
import os.path as osp
import pickle
import numpy as np
import scipy.sparse as sp

import torch
# from torch_scatter import scatter_add
from torch.utils.data import random_split

import utils
from pathlib import Path

class BaseDataset(object):
    def __init__(self, type: str, name: str, device: str = 'cpu'):
        
        path_parent = Path(__file__).parent.parent

        self.type = type
        self.name = name
        self.device = device
        if self.type in ['cocitation', 'coauthorship']:
            self.dataset_dir = osp.join('data', self.type, self.name) 
        else:
            self.dataset_dir = osp.join(path_parent, 'data', self.name)
        self.split_dir = osp.join(self.dataset_dir, 'splits')

        self.load_dataset()
        self.preprocess_dataset()

    def load_dataset(self):
        
        isNPZ = os.path.exists(os.path.join(self.dataset_dir, 'hypergraph.npz'))

        if isNPZ:
            self.load_npz()
        else:
            with open(osp.join(self.dataset_dir, 'features.pickle'), 'rb') as f:
                self.features = pickle.load(f)
                # print(f'features {type(self.features)} {self.features.shape}')
            with open(osp.join(self.dataset_dir, 'hypergraph.pickle'), 'rb') as f:
                self.hypergraph = pickle.load(f)
                # print(f'hypergraph {type(self.hypergraph)} {len(self.hypergraph)}')
            with open(osp.join(self.dataset_dir, 'labels.pickle'), 'rb') as f:
                self.labels = pickle.load(f)
                # print(f'labels {type(self.labels)} {len(self.labels)}')

    def load_npz(self):
        self.features = sp.load_npz(os.path.join(self.dataset_dir, 'features.npz')).astype(np.float32)
        # print(f'features {type(self.features)} {self.features.shape}')

        hg_adj = sp.load_npz(os.path.join(self.dataset_dir, 'hypergraph.npz'))
        np.clip(hg_adj.data, 0, 1, out = hg_adj.data)
        self.hypergraph = {}
        for index, edge in enumerate(hg_adj):
            self.hypergraph[index] = set(edge.indices)
        # print(f'hypergraph {type(self.hypergraph)} {len(self.hypergraph)}')
        
        self.labels = np.load(os.path.join(self.dataset_dir, 'labels.npy')).tolist()
        # print(f'labels {type(self.labels)} {len(self.labels)}')

    def load_splits(self, seed: int):
        with open(osp.join(self.split_dir, f'{seed}.pickle'), 'rb') as f:
            splits = pickle.load(f)
        return splits
    
    def preprocess_dataset(self):
        
        edge_set = set(self.hypergraph.keys())
        edge_to_num = {}
        num_to_edge = {}
        num = 0
        for edge in edge_set:
            edge_to_num[edge] = num
            num_to_edge[num] = edge
            num += 1

        inc = sp.lil_matrix((len(edge_set), self.features.shape[0]), dtype=np.int8)

        maxNodeID = -1
        nodeset = set()
        incidence_matrix = []
        processed_hypergraph = {}
        for edge in edge_set:
            nodes = self.hypergraph[edge]
            processed_hypergraph[edge_to_num[edge]] = nodes
            
            # inc[edge_to_num[edge], nodes] = 1
            for node in nodes:
                maxNodeID = max(maxNodeID, node)
                nodeset.add(node)
                inc[edge_to_num[edge], node] = 1
                incidence_matrix.append([node, edge_to_num[edge]])
        incident = inc.tocsr()
        # print(f'maxNodeID {maxNodeID} {len(nodeset)}')
        
        # self.adjacency = utils.clique_reduction(incident, self.features.shape[0])
        # print(f'# of nodes {self.features.shape[0]} # of edges {self.adjacency.count_nonzero()}')

        # self.adjacency = utils.randomwalk_reduction(incident, self.features, self.dataset_dir)
        # print(f'# of nodes {self.features.shape[0]} {len(self.labels)} # of edges {self.adjacency.count_nonzero()}')

        self.adjacency = utils.MAHC_reduction(incident, self.features, self.dataset_dir)
        print(f'# of nodes {self.features.shape[0]} {len(self.labels)} # of edges {self.adjacency.count_nonzero()}')

        row, col, data = self.adjacency.row, self.adjacency.col, self.adjacency.data
        # data = np.sort(data)
        # print(f'min {np.min(data)} max {np.max(data)} mean {np.mean(data)}')
        # cutoff = data[int(len(data) * 0.4)]
        adjacency_matrix = []
        for i in range(len(row)):
            # if data[i] < cutoff:
            #     continue
            adjacency_matrix.append([row[i], col[i]])
        # adjacency_matrix = []
        # for i in range(len(self.adjacency.nonzero()[0])):
        #     adjacency_matrix.append([self.adjacency.nonzero()[0][i], self.adjacency.nonzero()[1][i]])
        self.dyadicedge_index = torch.LongTensor(adjacency_matrix).T.contiguous()
        self.num_dyadicedges = len(self.dyadicedge_index[0])
        # print(f'dyadicedge_index {self.dyadicedge_index.shape} {self.dyadicedge_index}')
        # print(f'num_dyadicedges {self.num_dyadicedges}')

        self.processed_hypergraph = processed_hypergraph
        self.features = torch.as_tensor(self.features.toarray())
        self.hyperedge_index = torch.LongTensor(incidence_matrix).T.contiguous()
        self.labels = torch.LongTensor(self.labels)
        self.num_nodes = self.features.shape[0]
        self.num_hyperedges = int(self.hyperedge_index[1].max()) + 1
        self.edge_to_num = edge_to_num
        self.num_to_edge = num_to_edge

        weight = torch.ones(self.num_hyperedges)
        # Dn = scatter_add(weight[self.hyperedge_index[1]], self.hyperedge_index[0], dim=0, dim_size=self.num_nodes)
        # De = scatter_add(torch.ones(self.hyperedge_index.shape[1]), self.hyperedge_index[1], dim=0, dim_size=self.num_hyperedges)
        
        # print(f'data loaded.')
        # print('=============== Dataset Stats ===============')
        # print(f'dataset type: {self.type}, dataset name: {self.name}')
        # print(f'features size: [{self.features.shape[0]}, {self.features.shape[1]}]')
        # print(f'num nodes: {self.num_nodes} {int(self.hyperedge_index[0].max()) + 1}')
        # print(f'num hyperedges: {self.num_hyperedges} {len(self.hypergraph)}')
        # print(f'num_dyadicedges {self.num_dyadicedges}')
        # print(f'num connections: {self.hyperedge_index.shape[1]}')
        # print(f'num classes: {int(self.labels.max()) + 1}')
        # print(f'avg hyperedge size: {torch.mean(De).item():.2f}+-{torch.std(De).item():.2f}')
        # print(f'avg hypernode degree: {torch.mean(Dn).item():.2f}+-{torch.std(Dn).item():.2f}')
        # print(f'max node size: {Dn.max().item()}')
        # print(f'max edge size: {De.max().item()}')
        # print('=============================================')

        self.to(self.device)

    def to(self, device: str):
        self.features = self.features.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.dyadicedge_index = self.dyadicedge_index.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self

    def generate_random_split(self, train_ratio: float = 0.1, val_ratio: float = 0.1,
                              seed: int = 0, use_stored_split: bool = True):
        if use_stored_split:
            splits = self.load_splits(seed)
            train_mask = torch.tensor(splits['train_mask'], dtype=torch.bool, device=self.device)
            val_mask = torch.tensor(splits['val_mask'], dtype=torch.bool, device=self.device)
            test_mask = torch.tensor(splits['test_mask'], dtype=torch.bool, device=self.device)

        else:
            num_train = int(self.num_nodes * train_ratio)
            num_val = int(self.num_nodes * val_ratio)
            num_test = self.num_nodes - (num_train + num_val)

            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
            else:
                generator = torch.default_generator

            train_set, val_set, test_set = random_split(
                torch.arange(0, self.num_nodes), (num_train, num_val, num_test), 
                generator=generator)
            train_idx, val_idx, test_idx = \
                train_set.indices, val_set.indices, test_set.indices
            train_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            val_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            test_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

            # dic = {}
            # dic['train_mask'] = train_mask
            # dic['val_mask'] = val_mask
            # dic['test_mask'] = test_mask
            # with open(osp.join(self.split_dir, f'{seed}.pickle'), 'rb') as f:
            #     splits = pickle.load(f)
            #     print(type(splits['train_mask']))
            #     train_mask = torch.tensor(splits['train_mask'], dtype=torch.bool, device=self.device)
            #     val_mask = torch.tensor(splits['val_mask'], dtype=torch.bool, device=self.device)
            #     test_mask = torch.tensor(splits['test_mask'], dtype=torch.bool, device=self.device)

            #     print(f'seed {seed}')
            #     print(torch.all(dic['train_mask'].eq(train_mask)))
            #     print(torch.all(dic['val_mask'].eq(val_mask)))
            #     print(torch.all(dic['test_mask'].eq(test_mask)))

            overwrite = True
            if overwrite:
                dic = {}
                dic['train_mask'] = train_mask.detach().cpu().numpy()
                dic['val_mask'] = val_mask.detach().cpu().numpy()
                dic['test_mask'] = test_mask.detach().cpu().numpy()
                with open(osp.join(self.split_dir, f'{seed}.pickle'), 'wb') as f:
                    pickle.dump(dic, f)

        return [train_mask, val_mask, test_mask]



class CoraCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'cora', **kwargs)


class CiteseerCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'citeseer', **kwargs)


class PubmedCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'pubmed', **kwargs)


class CoraCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('coauthorship', 'cora', **kwargs)


class DBLPCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('coauthorship', 'dblp', **kwargs)


class ZooDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', 'zoo', **kwargs)


class NewsDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', '20newsW100', **kwargs)


class MushroomDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', 'Mushroom', **kwargs)


class NTU2012Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'NTU2012', **kwargs)


class ModelNet40Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'ModelNet40', **kwargs)

class CociWikiDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'coci_wiki', **kwargs)

class CoauCoraDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'coau_cora', **kwargs)

class CoauDblpDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'coau_dblp', **kwargs)

class CociCoraDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'coci_cora', **kwargs)

class CociCiteataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'coci_citeseer', **kwargs)

class CociPubmtaset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'coci_pubmed', **kwargs)

class News20Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', '20news', **kwargs)

class CociPatentC13(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'coci_patent_C13', **kwargs)
        