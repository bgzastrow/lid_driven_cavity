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
    vector : (Nx*Ny,)
    Nx : int
    Ny : int
    """
    vector: np.ndarray
    Nx: int
    Ny: int

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
        return self.vector[j*self.Nx+i]

    @property
    def matrix(self):
        """
        Returns 2D array version of state vector.

        Returns
        -------
        matrix : (n,n)
            2D ndarray for contour plotting, etc, with origin at bottom left
            of grid. Note that this may need to be manipulated for certain
            plotting functions.
        """
        # # convert to 2D
        # matrix = self.vector.reshape((self.Ny, self.Nx))

        # # flip vertically (flip rows)
        # matrix = np.flip(matrix, axis=0)

        # return matrix
        return get_matrix(self.vector, self.Nx, self.Ny)


def compute_u_hat(state, direction):
    """
    Spatially interpolates (averages) a state across a 2D grid.

    Parameters
    ----------
    state : State
        Representing u or v on a 2D grid.
    direction : str
        'x' or 'y'

    Returns
    -------
    state_hat : State
        Interpolated state, direction of grid is reduced by 1 in "direction".
    """
    if direction == 'x':
        interpolated_matrix = (state.matrix[:, :-1] + state.matrix[:, 1:]) / 2.0
        return interpolated_matrix
        # return State(
        #     get_vector(interpolated_matrix),
        #     Nx=state.Nx-1,
        #     Ny=state.Ny,
        #     )
    if direction == 'y':
        interpolated_matrix = (state.matrix[:-1, :] + state.matrix[1:, :]) / 2.0
        return interpolated_matrix
        # return State(
        #     get_vector(interpolated_matrix),
        #     Nx=state.Nx,
        #     Ny=state.Ny-1,
        #     )
    raise ValueError(f'direction = {direction:s} must equal "x" or "y" ')


def get_vector(matrix):
    """Converts a matrix in State format back to a state vector."""
    return np.flip(matrix, axis=0).reshape(-1,)


def get_matrix(vector, Nx, Ny):
    """Converts a vector in State format back to state matrix."""
    matrix = vector.reshape((Ny, Nx))  # convert to 2D
    matrix = np.flip(matrix, axis=0)  # flip vertically (flip rows)
    return matrix
