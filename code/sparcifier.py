import numpy as np
from scipy.sparse import csr_matrix
from sknetwork.utils.format import directed2undirected
import scipy.sparse as sp
from spanning_tree import spanning_tree2

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
    
def pack_tree_sum0(M, n, tau):

    residual = -directed2undirected(M)
    nonzero, = residual.diagonal().nonzero()
    residual[nonzero, nonzero] = 0
    residual.eliminate_zeros()

    for i in range(tau):

        tree = residual.copy()
        spanning_tree2(tree)

        residual = residual - (tree + tree.T)

    # printMatrix(M)
    # printMatrix(residual)
    remain = M + residual
    # printMatrix(remain)
    remain[remain<0] = 0
    remain.eliminate_zeros()
    # printMatrix(remain)

    return remain 

