import random
from itertools import permutations

import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add


import scipy.sparse as sp
from scipy.special import comb as choose
import itertools
import igraph as ig

import os
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import diags
import sparcifier

def fix_seed(seed):
    pass
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.set_deterministic_debug_mode(0)


def drop_features(x: Tensor, p: float, seed: int):
    fix_seed(seed)
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]

def drop_incidence(hyperedge_index: Tensor, p: float = 0.2, seed: int=0):
    fix_seed(seed)
    if p == 0.0:
        return hyperedge_index
    
    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p

    # print(mask)
    
    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)

    # print(f'hyperedge_index {hyperedge_index.shape}')
    # print(f'hyperedge_index {hyperedge_index}')
    return hyperedge_index


def drop_nodes(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_nodes, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[drop_idx, :] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def drop_hyperedges(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_edges, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[:, drop_idx] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index

def drop_neighbor(neighbors: Tensor, p: float = 0.2):
    if p == 0.0:
        return neighbors
    
    print(neighbors.count_nonzero())

    row, col = neighbors
    mask = torch.rand(row.size(0), device=neighbors.device) >= p

    row, col, _ = filter_incidence(row, col, None, mask)
    neighbors = torch.stack([row, col], dim=0)
    print(neighbors.count_nonzero())

    return neighbors

def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_edges: int):
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def common_node_edge_mask(hyperedge_indexs: list[Tensor], num_nodes: int, num_edges: int):
    hyperedge_weight = hyperedge_indexs[0].new_ones(num_edges)
    node_mask = hyperedge_indexs[0].new_ones((num_nodes,)).to(torch.bool)
    edge_mask = hyperedge_indexs[0].new_ones((num_edges,)).to(torch.bool)

    for index in hyperedge_indexs:
        Dn = scatter_add(hyperedge_weight[index[1]], index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(index.new_ones(index.shape[1]), index[1], dim=0, dim_size=num_edges)
        node_mask &= Dn != 0
        edge_mask &= De != 0
    return node_mask, edge_mask


def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()
    return masked_hyperedge_index


def clique_reduction(incident, n):
    e = []
    w = []
    m = incident.shape[0]
    hyperedges = incident.tolil().rows
    for i in range(m):
        edge = hyperedges[i]
        den = choose(len(edge),2)
        s = [list(k) for k in itertools.combinations(edge,2)]
        # x = [1/den]*len(s)
        e.extend(s)
        # w.extend(x)

    # g = ig.Graph()
    # g.add_vertices(n)
    # g.add_edges([tuple(x) for x in e])
    # g.es["weight"] = w
    # g = g.simplify(combine_edges=sum)
    
    # matrix = np.array(g.get_adjacency().data)

    # return sp.csr_matrix(matrix).tocoo()
    return e

def randomwalk_reduction(incident, features, prepath):

    n = features.shape[0]

    dataset = prepath.split("/")[len(prepath.split("/")) - 1]

    approx_knn = False
    knn_k = 10

    knn_path = os.path.join(prepath, 'knn.npz')
    knn = None
    # check if file exist
    if os.path.isfile(knn_path):
        knn = sp.load_npz(knn_path)
        # print("read knn from file")
        knn.data = 1.0-knn.data
    else:
        if approx_knn:
            import scann
            ftd = features.todense()
            if dataset.startswith('amazon'):
                searcher = scann.scann_ops_pybind.load_searcher('scann_amazon')
            else:
                searcher = scann.scann_ops_pybind.load_searcher('scann_magpm')
            neighbors, distances = searcher.search_batched_parallel(ftd)
            del ftd
            knn = sp.csr_matrix(((distances.ravel()), neighbors.ravel(), np.arange(0, neighbors.size+1, neighbors.shape[1])), shape=(n, n))
            knn.setdiag(0.0)
            # save knn to npz file
            knn_path = os.path.join(prepath, 'knn_apprx.npz')
            sp.save_npz(knn_path, knn)
            # print("save apprx knn to file")
        else:
            knn = kneighbors_graph(features, knn_k, metric="cosine", mode="distance", n_jobs=16)
            # save knn to npz file
            sp.save_npz(knn_path, knn)
            # print("save knn to file")
            knn.data = 1.0-knn.data

    P_attr = normalize(knn + knn.T, norm='l1')

    P_topo = normalize(incident.T, norm='l1', axis=1)@(normalize(incident, norm='l1', axis=1))

    # print(f'P_topo {P_topo.shape} {P_topo.count_nonzero()}')

    # build tree 
    num_trees = 3
    residual = -P_topo
    remain = sp.csr_matrix((n, n))
    for i in range(num_trees):
        # print("round " + str(i))
        # tree = spanning_tree2(residual.copy(), overwrite=True)
        # tree = large_spanning_tree(residual.copy(), config.prepath, overwrite=True)
        tree = minimum_spanning_tree(residual.copy(), overwrite=True)
        residual = residual - tree
        remain = remain - tree
    P_topo = remain

    # print(f'P_topo {P_topo.shape} {P_topo.count_nonzero()}')

    # random walk
    alpha = 0.2
    f0 = alpha
    f1 = alpha * (1.0 - alpha)
    f2 = alpha * (1.0 - alpha)**2
    # P_topo = f0 * diags(np.ones(n)) + f1 * P_topo
    P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo

    # print(f'P_topo {P_topo.shape} {P_topo.count_nonzero()}')

    # PP = P_topo
    PP = P_topo@P_attr
    PP = normalize(PP, norm='l1')

    # x = x^1/2
    # print("x = power(1/2)")
    pw = 1.0 / 2.0
    PP.data = PP.data ** pw
    
    # print(type(PP))

    matrix = PP.tocoo()
    # print(type(matrix))
    print(f'matrix {matrix.shape} {matrix.count_nonzero()}')

    return matrix

def MAHC_reduction(incident, features, prepath):
    n = features.shape[0]

    dataset = prepath.split("/")[len(prepath.split("/")) - 1]

    approx_knn = False
    knn_k = 10

    knn_path = os.path.join(prepath, 'knn.npz')
    knn = None
    # check if file exist
    if os.path.isfile(knn_path):
        knn = sp.load_npz(knn_path)
        # print("read knn from file")
        knn.data = 1.0-knn.data
    else:
        if approx_knn:
            import scann
            ftd = features.todense()
            if dataset.startswith('amazon'):
                searcher = scann.scann_ops_pybind.load_searcher('scann_amazon')
            else:
                searcher = scann.scann_ops_pybind.load_searcher('scann_magpm')
            neighbors, distances = searcher.search_batched_parallel(ftd)
            del ftd
            knn = sp.csr_matrix(((distances.ravel()), neighbors.ravel(), np.arange(0, neighbors.size+1, neighbors.shape[1])), shape=(n, n))
            knn.setdiag(0.0)
            # save knn to npz file
            knn_path = os.path.join(prepath, 'knn_apprx.npz')
            sp.save_npz(knn_path, knn)
            # print("save apprx knn to file")
        else:
            knn = kneighbors_graph(features, knn_k, metric="cosine", mode="distance", n_jobs=16)
            # save knn to npz file
            sp.save_npz(knn_path, knn)
            # print("save knn to file")
            knn.data = 1.0-knn.data

    P_attr = normalize(knn + knn.T, norm='l1')

    P_topo = normalize(incident.T, norm='l1', axis=1)@(normalize(incident, norm='l1', axis=1))

    # build tree 
    num_trees = 3
    P_topo = sparcifier.pack_tree_sum0(P_topo, n, num_trees)

    # random walk
    alpha = 0.2
    gamma = 2

    f0 = alpha
    f1 = alpha * (1.0 - alpha)
    f2 = alpha * (1.0 - alpha)**2
    f3 = alpha * (1.0 - alpha)**3

    if gamma == 1:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo
    elif gamma == 2:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo
    elif gamma == 3:
        P_topo = f0 * diags(np.ones(n)) + f1 * P_topo + f2 * P_topo@P_topo + f3 * P_topo@P_topo@P_topo

    P_topo = normalize(P_topo, norm='l1')
    
    print("no norm")
    PP_sum = P_topo@P_attr

    #PP_sum = normalize(P_topo@P_attr, norm='l1')

    # x = x^1/2
    # print("x = power(1/2)")
    pw = 1.0 / 2.0
    PP_sum.data = PP_sum.data ** pw

    matrix = PP_sum.tocoo()
    # print(f'matrix {matrix.shape} {matrix.count_nonzero()}')

    return matrix