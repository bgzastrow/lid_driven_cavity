"""Contains logic for solving the lid-driven cavity flow problem."""
from dataclasses import dataclass, field
from typing import List
import numpy as np
import scipy as sp


@dataclass
class Operator:
    """
    Sparse linear algebra representation of discretized spatial operator.

    Also contains logic to convert to CSR array format or print in dense form.

    Parameters
    ----------
    shape : tuple
        Dimensions of operator. Need to specify explicitly because of sparse
        representation format.
    data : list of float
        Values of each nonzero entry in operator matrix.
    row : list of int
        Row index of each nonzero entry in operator matrix.
    col : list of int
        Column index of each nonzero entry in operator matrix.
    """
    shape: tuple
    data: List[float] = field(default_factory=list)
    row: List[int] = field(default_factory=list)
    col: List[int] = field(default_factory=list)

    def append(self, row, col, data):
        """
        Inserts a new nonzero entry to the matrix.

        i.e. executes the following operation: matrix[row, col] = data.

        Parameters
        ----------
        row : int
            Nonzero entry in operator matrix.
        col : int
            Row index of nonzero entry.
        data : float
            Column index of nonzero entry.
        """
        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    def get_csr(self):
        """
        Builds COO matrix and converts it to CSR.

        Returns
        -------
        matrix_csr : sp.sparse.csr_array
            Sparse operator matrix in CSR form.
        """
        matrix_coo = sp.sparse.coo_array(
            (self.data, (self.row, self.col)),
            shape=self.shape,
            dtype=np.double,
        )
        matrix_csr = sp.sparse.csr_array(matrix_coo)
        return matrix_csr

    def print_dense(self):
        """
        Prints operator matrix in dense form for visualization.
        """
        matrix_csr = self.get_csr()
        print(matrix_csr.toarray())

    def multiply(self, x):
        """
        Computes the matrix-vector product of the operator with a vector x.

        Parameters
        ----------
        x : (self.shape[1],)
            Vector to be multiplied.

        Returns
        -------
        Ax : (self.shape[0],)
            Vector result of matrix-vector product.
        """
        matrix_csr = self.get_csr()
        return matrix_csr.dot(x)
