import numpy as np
import time
from scipy.sparse import coo_matrix, csgraph
from sknetwork.utils.format import directed2undirected
import scipy.sparse as sp
from spanning_tree import spanning_tree2
# from spanning_tree_0 import spanning_tree2

def check_symmetric(A, rtol=1e-08, atol=1e-08):
    AT = A.T
    for i in range(len(A)):
        for j in range(len(A[0])):
            if abs(AT[i][j] - A[i][j]) > rtol:
                return False
    return True

def printMatrix(A):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    A_arr = A.toarray()
    mask = np.arange(10) % 10 < 10
    result = A_arr[np.ix_(mask, mask)]
    print(result)
    
def pack_tree_sum(M, n, tau):

    # start = time.time()
    # M.setdiag(0)
    nonzero, = M.diagonal().nonzero()
    M[nonzero, nonzero] = 0
    M.eliminate_zeros()

    residual = -M - M.T
    # print(time.time() - start)

    # times = [0 for _ in range(2)]

    for i in range(tau):

        # start = time.time()

        tree = residual.copy()
        spanning_tree2(tree)

        # times[0] += time.time() - start

        # start = time.time()
        
        # tree = tree + tree.T
        residual = residual - (tree + tree.T)

        # times[1] += time.time() - start

        # remain = remain - tree

    # for i in range(len(times)):
    #     print(f'i {i} {(times[i])}')
    
    # start = time.time()
    MM = M + residual
    MM[MM<0] = 0
    MM.eliminate_zeros()
    # print(time.time() - start)

    return MM 


def pack_tree_sum0(M, n, tau):

    residual = -directed2undirected(M)
    nonzero, = residual.diagonal().nonzero()
    residual[nonzero, nonzero] = 0
    residual.eliminate_zeros()

    for i in range(tau):

        tree = residual.copy()
        spanning_tree2(tree)

        residual -= (tree + tree.T)

    remain = M + residual
    remain[remain<0] = 0
    remain.eliminate_zeros()
    # print(remain.count_nonzero())

    return remain 

def pack_tree_sym(M, n, tau):

    residual = -directed2undirected(M.copy())
    remain = sp.spdiags(M.diagonal() * 2, 0, n, n)
    
    for i in range(tau):
        tree = spanning_tree2(residual.copy(), overwrite=True)
        # tree = sp.csgraph.minimum_spanning_tree(residual.copy(), overwrite=True)
        tree = tree + tree.T
        residual = residual - tree
        remain = remain - tree

    return remain 

def pack_tree_min(M, n, tau):

    residual = -M
    remain = sp.csr_matrix((n, n))
    # print(check_symmetric(residual.toarray()))

    for i in range(tau):

        tree = spanning_tree2(residual.copy(), overwrite=True)
        # tree = sp.csgraph.minimum_spanning_tree(residual.copy(), overwrite=True)
        # tree = tree + tree.T
        # print(check_symmetric(tree.toarray()))
        # print(check_symmetric(residual.toarray()))

        # print(f'tree symmetric {check_symmetric(tree.toarray())}')
        residual = residual - tree
        # print(f'residual symmetric {check_symmetric(residual.toarray())}')
        remain = remain - tree
        # print(f'remain symmetric {check_symmetric(remain.toarray())}')
        # print(f'{np.count_nonzero(remain.toarray())} {n}')

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    remain = remain + remain.T
    remain_arr = remain.toarray()
    mask = np.arange(10) % 10 < 10
    result = remain_arr[np.ix_(mask, mask)]
    # print(result)
    return remain 

def tree(M, n, tau):

    residual = -M
    remain = sp.csr_matrix((n, n))
    for i in range(tau):
        tree = spanning_tree2(residual.copy(), overwrite=True)
        residual = residual - tree
        remain = remain - tree

    return remain

def independent_sampling(M, tau, seed):

    M = M.tocoo()
    data = M.data
    row = M.row
    col = M.col

    n = len(M.toarray())
    m = len(row)

    np.random.seed(seed)
    size = min(2*tau*n, m)
    idx = np.random.choice(m, size, replace=False)

    data_ = data[idx]
    row_ = row[idx]
    col_ = col[idx]
    print(len(data_))

    M_ = coo_matrix((data_, (row_, col_)), shape=(n, n))

    return M_.tocsr(copy=False)


