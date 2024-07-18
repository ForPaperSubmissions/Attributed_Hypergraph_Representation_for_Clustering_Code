from scipy.sparse import csr_matrix

def mul_sparse_matrix(A, other):
    # Check if the dimensions are compatible for multiplication
    if A.shape[1] != other.shape[0]:
        raise ValueError("Matrix dimensions are not aligned for multiplication")

    # Perform sparse matrix multiplication
    result_data = []
    result_indices = []
    result_indptr = [0]

    # Iterate over rows of the first matrix (self)
    for i in range(A.shape[0]):
        # Extract the current row of the first matrix
        row = A.data[i]
        row_indices = A.indices[A.indptr[i]:A.indptr[i+1]]

        # Iterate over columns of the second matrix (other)
        for j in range(other.shape[1]):
            # Extract the current column of the second matrix
            col = other.data[j]
            col_indices = other.indices[other.indptr[j]:other.indptr[j+1]]

            # Compute the dot product between the row and column
            

            dot_product = sum(row[row_indices == col_indices] * col)
            
            # If the dot product is non-zero, add it to the result
            if dot_product != 0:
                result_data.append(dot_product)
                result_indices.extend([j] * len(row_indices))
        
        # Update the indptr for the next row
        result_indptr.append(len(result_data))

    # Create a new sparse matrix from the result data, indices, and indptr
    result_shape = (A.shape[0], other.shape[1])
    result_matrix = csr_matrix((result_data, result_indices, result_indptr), shape=result_shape)

    return result_matrix