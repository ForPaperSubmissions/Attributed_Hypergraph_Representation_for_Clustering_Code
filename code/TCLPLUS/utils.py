import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor

import scipy.sparse as sp

import os
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from scipy.sparse import diags

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import sparcifier

def drop_features(x: Tensor, p: float, seed: int):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]

def drop_incidence(hyperedge_index: Tensor, p: float = 0.2, seed: int=0):
    if p == 0.0:
        return hyperedge_index
    
    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p
    
    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)

    return hyperedge_index

def AHRC_reduction(incident, features, prepath):
    n = features.shape[0]

    dataset = prepath.split("/")[len(prepath.split("/")) - 1]

    approx_knn = False
    knn_k = 10

    knn_path = os.path.join(prepath, 'knn.npz')
    knn = None
    # check if file exist
    if os.path.isfile(knn_path):
        knn = sp.load_npz(knn_path)
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
        else:
            knn = kneighbors_graph(features, knn_k, metric="cosine", mode="distance", n_jobs=16)
            # save knn to npz file
            sp.save_npz(knn_path, knn)
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
    
    PP_sum = P_topo@P_attr

    # x = x^1/2
    pw = 1.0 / 2.0
    PP_sum.data = PP_sum.data ** pw

    matrix = PP_sum.tocoo()

    return matrix