import numpy as np
import time
import heapq, os
cimport numpy as np
cimport cython

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph._validation import validate_graph
from libc.stdio cimport printf

# python setup.py build_ext --inplace

np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

# Fused type for int32 and int64
ctypedef fused int32_or_int64:
    np.int32_t
    np.int64_t

# Another copy of the same fused type, for working with mixed-type functions.
ctypedef fused int32_or_int64_b:
    np.int32_t
    np.int64_t

# NULL_IDX is the index used in predecessor matrices to store a non-path
DEF NULL_IDX = -9999

def spanning_tree(csgraph, overwrite=False):

    global NULL_IDX
    
    cdef int N = csgraph.shape[0]

    # Stable sort is a necessary but not sufficient operation
    # to get to a canonical representation of solutions.

    i_sort = np.argsort(csgraph.data, kind='stable').astype(ITYPE)

    rank = np.zeros(N, dtype=ITYPE)
    predecessors = np.arange(N, dtype=ITYPE)
    row_indices = np.zeros(len(csgraph.data), dtype=ITYPE)

    _min_spanning_tree(csgraph.data, csgraph.indices, csgraph.indptr, i_sort,
                       row_indices, predecessors, rank)

    csgraph.eliminate_zeros()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _min_spanning_tree(DTYPE_t[::1] data,
                             ITYPE_t[::1] col_indices,
                             ITYPE_t[::1] indptr,
                             ITYPE_t[::1] i_sort,
                             ITYPE_t[::1] row_indices,
                             ITYPE_t[::1] predecessors,
                             ITYPE_t[::1] rank) nogil:
    # Work-horse routine for computing minimum spanning tree using
    #  Kruskal's algorithm.  By separating this code here, we get more
    #  efficient indexing.
    cdef unsigned int i, j, V1, V2, R1, R2, n_edges_in_mst, n_verts, n_data
    n_verts = predecessors.shape[0]
    n_data = i_sort.shape[0]

    # Arrange `row_indices` to contain the row index of each value in `data`.
    # Note that the array `col_indices` already contains the column index.
    for i in range(n_verts):
        for j in range(indptr[i], indptr[i + 1]):
            row_indices[j] = i

    # step through the edges from smallest to largest.
    #  V1 and V2 are connected vertices.
    n_edges_in_mst = 0
    i = 0
    while i < n_data and n_edges_in_mst < n_verts - 1:
        j = i_sort[i]
        # printf("data %d ranked in %d\n", data[j], i)
        V1 = row_indices[j]
        V2 = col_indices[j]

        # progress upward to the head node of each subtree
        R1 = V1
        while predecessors[R1] != R1:
            R1 = predecessors[R1]
        R2 = V2
        while predecessors[R2] != R2:
            R2 = predecessors[R2]

        # Compress both paths.
        while predecessors[V1] != R1:
            predecessors[V1] = R1
        while predecessors[V2] != R2:
            predecessors[V2] = R2
            
        # if the subtrees are different, then we connect them and keep the
        # edge.  Otherwise, we remove the edge: it duplicates one already
        # in the spanning tree.
        if R1 != R2:
            n_edges_in_mst += 1
            
            # Use approximate (because of path-compression) rank to try
            # to keep balanced trees.
            if rank[R1] > rank[R2]:
                predecessors[R2] = R1
            elif rank[R1] < rank[R2]:
                predecessors[R1] = R2
            else:
                predecessors[R2] = R1
                rank[R1] += 1
        else:
            data[j] = 0
        
        i += 1
        
    # We may have stopped early if we found a full-sized MST so zero out the rest
    while i < n_data:
        j = i_sort[i]
        data[j] = 0
        i += 1
