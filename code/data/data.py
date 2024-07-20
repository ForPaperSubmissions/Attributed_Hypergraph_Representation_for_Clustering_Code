import sys, os, pickle
import numpy as np, scipy.sparse as sp
import pickle
import inspect

sys.path.append('../')

path_to_data = ""

def load(dataset):
    
    # path to the dataset
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    path_to_data = os.path.join(current, dataset)

    isNPZ = os.path.exists(os.path.join(path_to_data, 'hypergraph.npz'))

    if isNPZ:
        data_dict = load_npz(dataset)
    else:
        ps = parser(dataset)

        with open(os.path.join(path_to_data, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

        with open(os.path.join(path_to_data, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle)
    
        with open(os.path.join(path_to_data, 'labels.pickle'), 'rb') as handle:
            labels = ps._1hot(pickle.load(handle))

        adj = sp.lil_matrix((len(hypergraph), features.shape[0]), dtype=np.int8)
        for index, edge in enumerate(hypergraph):
            hypergraph[edge] = list(hypergraph[edge])
            adj[index, hypergraph[edge]] = 1
        adj_sp = adj.tocsr()

        # print("# of keywords " + str(features.shape[1]) + " # of hyperedge " + str(len(hypergraph)) + " # of nodes " + str(features.shape[0]) + " average node degree " + f"{adj_sp.sum(0).mean():.1f}") 

        data_dict = {'hypergraph': hypergraph, 'features':features, 'labels': labels, 'name': dataset, 'adj': adj_sp}
    return data_dict

def load_npz(dataset):

    # m*n matrix
    hg_adj = sp.load_npz(os.path.join(path_to_data, 'hypergraph.npz'))
    np.clip(hg_adj.data, 0, 1, out=hg_adj.data)
    features = sp.load_npz(os.path.join(path_to_data, 'features.npz'))
    labels = np.load(os.path.join(path_to_data, 'labels.npy'))

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

    return {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'name': dataset, 'adj': hg_adj}

class parser(object):

    def __init__(self, dataset):

        self.dataset = dataset

    def parse(self):
        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()

    def _load_data(self):
        
        with open(os.path.join(path_to_data, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

        with open(os.path.join(path_to_data, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle).todense()

        with open(os.path.join(path_to_data, 'labels.pickle'), 'rb') as handle:
            labels = self._1hot(pickle.load(handle))

        return {'hypergraph': hypergraph, 'features': features, 'labels': labels}

    def _1hot(self, labels):
        classes = set(labels)
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)