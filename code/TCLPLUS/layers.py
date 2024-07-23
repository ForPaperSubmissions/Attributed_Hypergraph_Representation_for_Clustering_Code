from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros

import numpy as np
import random

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

class ProposedConv(MessagePassing):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0, 
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False, 
                 row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        # print(f'in_dim {in_dim} hid_dim {hid_dim} out_dim {out_dim}')
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm

        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')
        self.lin_n2n = Linear(in_dim, out_dim, bias=False, weight_initializer='glorot')

        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
            self.bias_n2n = Parameter(torch.Tensor(out_dim))

        else:
            self.register_parameter('bias_n2e', None) 
            self.register_parameter('bias_e2n', None) 
            self.register_parameter('bias_n2n', None) 
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # fix_seed(0)
        # torch.nn.init.kaiming_uniform_(self.lin_n2e.weight)
        # torch.nn.init.kaiming_uniform_(self.lin_e2n.weight)
        # torch.nn.init.kaiming_uniform_(self.lin_n2n.weight)

        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        self.lin_n2n.reset_parameters()

        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        zeros(self.bias_n2n)
    
    def forward(self, x: Tensor, hyperedge_index: Tensor, dyadicedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None, num_dyadicedges: Optional[int] = None):
        # print(f'ProposedConv forward')
        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        if num_dyadicedges is None and dyadicedge_index.numel() > 0:
            num_dyadicedges = len(dyadicedge_index[0])

        hyperedge_weight = x.new_ones(num_edges)
        dyadicedge_weight = x.new_ones(num_dyadicedges)

        node_idx, edge_idx = hyperedge_index
        dyadic_node1_idx, dyadic_node2_idx = dyadicedge_index

        Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                            hyperedge_index[1], dim=0, dim_size=num_edges)
        dyadic_Dn = scatter_add(dyadicedge_weight[dyadicedge_index[1]],
                            dyadicedge_index[0], dim=0, dim_size=num_nodes)

        if self.row_norm:
            Dn_inv = 1.0 / Dn
            Dn_inv[Dn_inv == float('inf')] = 0
            De_inv = 1.0 / De
            De_inv[De_inv == float('inf')] = 0
            dyadic_Dn_inv = 1.0 / dyadic_Dn
            dyadic_Dn_inv[dyadic_Dn_inv == float('inf')] = 0

            norm_n2e = De_inv[edge_idx]
            norm_e2n = Dn_inv[node_idx]
            norm_n2n = dyadic_Dn_inv[dyadic_node2_idx]
            
        else:
            Dn_inv_sqrt = Dn.pow(-0.5)
            Dn_inv_sqrt[Dn_inv_sqrt == float('inf')] = 0
            De_inv_sqrt = De.pow(-0.5)
            De_inv_sqrt[De_inv_sqrt == float('inf')] = 0
            dyadic_Dn_inv_sqrt = dyadic_Dn.pow(-0.5)
            dyadic_Dn_inv_sqrt[dyadic_Dn_inv_sqrt == float('inf')] = 0

            norm = De_inv_sqrt[edge_idx] * Dn_inv_sqrt[node_idx]
            
            norm_n2e = norm
            norm_e2n = norm

            norm_n2n = dyadic_Dn_inv_sqrt[dyadic_node2_idx] * dyadic_Dn_inv_sqrt[dyadic_node1_idx]

        # layer 2
        # node to node
        xx = self.lin_n2n(x)
        dyadic_n = self.propagate(dyadicedge_index, x=xx, norm=norm_n2n, 
                            size=(num_nodes, num_nodes))  # Node to node
        
        if self.bias_n2n is not None:
            dyadic_n = dyadic_n + self.bias_n2n
        # dyadic_n = self.act(dyadic_n)
        # dyadic_n = F.dropout(dyadic_n, p=self.dropout, training=self.training)

        # print(f'before x {x.shape}')
        x = self.lin_n2e(x)
        # print(f'after x {x.shape}')
        # print(f'norm_n2e {norm_n2e} {norm_n2e.shape}')
        # print(f'hyperedge_index {hyperedge_index} {hyperedge_index.shape}')
        e = self.propagate(hyperedge_index, x=x, norm=norm_n2e, 
                               size=(num_nodes, num_edges))  # Node to edge
        # print(f'after propagate e {e.shape}') 

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=self.dropout, training=self.training)

        # print(f'before x {x.shape}')
        x = self.lin_e2n(e)
        # print(f'after x {x.shape}')
        n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n, 
                               size=(num_edges, num_nodes))  # Edge to node
        # print(f'after propagate n {n.shape}') 
        
        if self.bias_e2n is not None:
            n = n + self.bias_e2n

        return n, e, dyadic_n # No act, act

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j
