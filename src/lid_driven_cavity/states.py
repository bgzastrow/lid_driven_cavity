"""Contains State class."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class State:
    """
    Represents a State object for a single state type (u, v, or phi).

    Parameters
    ----------
    entries : np.ndarray
        vector or matrix representation of state
    Nx : int
        number of grid points in x direction
    Ny : int
        number of grid points in y direction
    """
    entries: np.ndarray
    Nx: int
    Ny: int
    vector: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.entries.ndim == 1:
            self.vector = self.entries
        elif self.entries.ndim == 2:
            self.vector = matrix_to_vector(self.entries)

    def ij(self, i, j):
        """
        Returns the requested (i,j)-indexed value for computation.

        Parameters
        ----------
        i : int
            x index of state in 2d grid form.
        j : int
            y index of state in 2d grid form.

        Returns
        -------
        value : float
            Value at (i,j) index of state in 2d grid form, with origin at
            bottom left of grid. Counts from bottom to top and from left to
            right.
        """
        return self.vector[lij(i, j, self.Nx)]

    def set_matrix(self, matrix):
        """Update state using matrix form."""
        self.vector = matrix_to_vector(matrix)

    def get_matrix(self):
        """Get state in matrix form."""
        return vector_to_matrix(self.vector, Nx=self.Nx, Ny=self.Ny)

    def __repr__(self):
        """Returns string representation of State."""
        return np.array_str(self.get_matrix())

    def pad_boundaries(self):
        """Pads state with zero boundaries."""
        padded_matrix = pad_boundaries(self.get_matrix())
        self.set_matrix(padded_matrix)
        self.Nx = self.Nx + 2
        self.Ny = self.Ny + 2

    def strip_boundaries(self):
        """Strips boundaries from state."""
        stripped_matrix = strip_boundaries(self.get_matrix())
        self.set_matrix(stripped_matrix)
        self.Nx = self.Nx - 2
        self.Ny = self.Ny - 2


def matrix_to_vector(matrix):
    """Converts a matrix in State format to a State vector."""
    return np.flip(matrix, axis=0).reshape(-1,)


def vector_to_matrix(vector, Nx, Ny):
    """Converts a vector in State format to State matrix."""
    return np.flip(vector.reshape((Ny, Nx)), axis=0)  # 2D then flip vertically


def lij(i, j, Nx):
    """Row major ordering indexing function."""
    return j * Nx + i


def pad_boundaries(matrix):
    """
    Adds a border of zeros around an existing matrix.

    Increases x and y dimensions of array by 2 each.

    Parameters
    ----------
    matrix : np.ndarray (Nx, Ny)
        initial matrix to be padded with zeros.
    Returns
    -------
    padded_matrix : np.ndarray (Nx+2, Ny+2)
        original matrix with a border of zero entries added.
    """
    matrix_padded = np.zeros((matrix.shape[0]+2, matrix.shape[1]+2))
    matrix_padded[1:-1, 1:-1] = matrix
    return matrix_padded


def strip_boundaries(matrix):
    """
    Removes the outermost row/column from each boundary of an existing matrix.

    Reduces the x and y dimensions of an array by 2 each.

    Parameters
    ----------
    matrix : np.ndarray (Nx, Ny)
        initial matrix to be stripped of its boundary rows and columns.

    Returns
    -------
    stripped_matrix : np.ndarray (Nx-2, Ny-2)
        original matrix with its boundary rows and columns removed
    """
    return matrix[1:-1, 1:-1]
