import sys, os, pickle
import numpy as np, scipy.sparse as sp
import pickle
import inspect

sys.path.append('../')
import config

def load(dataset):

    # remeber dataset
    config.dataset = dataset
    
    # path to the dataset
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    config.prepath = os.path.join(current, dataset)

    isNPZ = os.path.exists(os.path.join(config.prepath, 'hypergraph.npz'))

    if isNPZ:
        data_dict = load_npz(dataset)
    else:
        ps = parser(dataset)

        with open(os.path.join(config.prepath, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

        with open(os.path.join(config.prepath, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle)
    
        with open(os.path.join(config.prepath, 'labels.pickle'), 'rb') as handle:
            labels = ps._1hot(pickle.load(handle))

        adj = sp.lil_matrix((len(hypergraph), features.shape[0]), dtype=np.int8)
        for index, edge in enumerate(hypergraph):
            hypergraph[edge] = list(hypergraph[edge])
            adj[index, hypergraph[edge]] = 1
        adj_sp = adj.tocsr()

        print("# of keywords " + str(features.shape[1]) + " # of hyperedge " + str(len(hypergraph)) + " # of nodes " + str(features.shape[0]) + " average node degree " + f"{adj_sp.sum(0).mean():.1f}") 

        data_dict = {'hypergraph': hypergraph, 'features': features.todense(), 'features_sp':features, 'labels': labels, 'n': features.shape[0], 'e': len(hypergraph), 'name': dataset, 'adj': adj, 'adj_sp': adj_sp}
    return data_dict

def load_npz(dataset):
    
    # hg_adj = sp.load_npz(f'data/npz/{dataset}/hypergraph.npz')
    # np.clip(hg_adj.data, 0, 1, out=hg_adj.data)
    # features = sp.load_npz(f'data/npz/{dataset}/features.npz')
    # labels = np.load(f'data/npz/{dataset}/labels.npy')

    # m*n matrix
    hg_adj = sp.load_npz(os.path.join(config.prepath, 'hypergraph.npz'))
    np.clip(hg_adj.data, 0, 1, out=hg_adj.data)
    features = sp.load_npz(os.path.join(config.prepath, 'features.npz'))
    labels = np.load(os.path.join(config.prepath, 'labels.npy'))

    for row in range(features.shape[0]):
        data_indices = features.indices[features.indptr[row]:features.indptr[row+1]]
        data_values = features.data[features.indptr[row]:features.indptr[row+1]]
        
        for i in range(len(data_values)):
            if data_values[i] < 0:
                print(f"Row {row}, Column {data_indices[i]}: {data_values[i]}")

    # a collection of hyperedges, each of which is a set of nodes
    hypergraph = {}
    for index, edge in enumerate(hg_adj):
        hypergraph[index] = list(edge.indices)

    print("# of keywords " + str(features.shape[1]) + " # of hyperedge " + str(hg_adj.shape[0]) + " # of nodes " + str(features.shape[0]) + " average node degree " + f"{hg_adj.sum(0).mean():.1f}")

    return {'hypergraph': hypergraph, 'features': features.todense(), 'features_sp': features, 'labels': labels, 'n': features.shape[0], 'e': hg_adj.shape[0], 'name': dataset, 'adj': hg_adj, 'adj_sp': hg_adj}

class parser(object):

    def __init__(self, dataset):

        self.dataset = dataset

    def parse(self):
        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()

    def _load_data(self):
        
        with open(os.path.join(config.prepath, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

        with open(os.path.join(config.prepath, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle).todense()

        with open(os.path.join(config.prepath, 'labels.pickle'), 'rb') as handle:
            labels = self._1hot(pickle.load(handle))

        return {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0]}

    def _1hot(self, labels):
        classes = set(labels)
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)