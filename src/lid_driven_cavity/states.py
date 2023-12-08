"""Contains State class."""
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class State:
    """
    Single state vector for state values across entire grid. Assumes square
    grid!

    Parameters
    ----------
    vector : (N^2,)
    """
    vector: np.ndarray
    N: int = field(init=False)

    def __post_init__(self):
        """Gets N and checks that dim(vector) is a perfect square."""
        self.N = int(math.sqrt(self.vector.shape[0]))
        if self.N**2 != self.vector.shape[0]:
            raise ValueError(f"vector.shape[0]={self.vector.shape[0]:d}"
                             " must be a perfect square")

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
        return self.vector[j*self.N+i]

    def get_matrix(self):
        """
        Returns 2D array version of state vector.

        Returns
        -------
        matrix : (n,n)
            2D ndarray for contour plotting, etc, with origin at bottom left
            of grid. Note that this may need to be manipulated for certain
            plotting functions.
        """
        # convert to 2D
        matrix = self.vector.reshape((self.N, self.N))

        # flip vertically (flip rows)
        matrix = np.flip(matrix, axis=0)

        return matrix
