"""Tests the State class functionality."""
import pytest
import numpy as np

from lid_driven_cavity.states import State, interpolate, get_vector


@pytest.fixture(name='i', params=range(5))
def create_mock_i(request):
    """Mock int value(s)."""
    return request.param


@pytest.fixture(name='j', params=range(4))
def create_mock_j(request):
    """Mock int value(s)."""
    return request.param


def test_get_matrix():
    """Tests that the correct matrix is created from the vector."""
    Nx, Ny = 5, 4
    state = State(np.arange(0, Nx*Ny), Nx=Nx, Ny=Ny)
    mock_matrix = np.array([
        [15, 16, 17, 18, 19],
        [10, 11, 12, 13, 14],
        [5,   6,  7,  8,  9],
        [0,   1,  2,  3,  4],
    ])
    assert mock_matrix == pytest.approx(state.matrix)


def test_matrix_equals_ij(i, j):
    """Tests that ij indexing function matches matrix output."""
    Nx, Ny = 5, 4
    state = State(np.arange(0, Nx*Ny), Nx=Nx, Ny=Ny)
    matrix = state.matrix
    assert state.ij(i, j) == pytest.approx(matrix[state.Ny-j-1, i])


def test_get_vector_x():
    """
    Converts a matrix in State format back to a state vector.

    Create a state vector.
    Create a State object from the state vector.
    Get the matrix version of the State.
    Convert that matrix back to a vector again.
    Compare to ensure no changes.
    """
    vector = np.arange(5*4)
    state = State(vector, 5, 4)
    matrix = state.matrix
    new_vector = get_vector(matrix)
    assert vector == pytest.approx(new_vector)


def test_get_vector_y():
    """
    Converts a matrix in State format back to a state vector.

    Create a state vector.
    Create a State object from the state vector.
    Get the matrix version of the State.
    Convert that matrix back to a vector again.
    Compare to ensure no changes.
    """
    vector = np.arange(4*5)
    state = State(vector, 4, 5)
    matrix = state.matrix
    new_vector = get_vector(matrix)
    assert vector == pytest.approx(new_vector)


def test_interpolate_x():
    """Tests that the interpolation works in the x-direction."""
    Nx, Ny = 5, 4
    state = State(np.arange(Nx*Ny), Nx=Nx, Ny=Ny)
    mock = np.array([
        [15.5, 16.5, 17.5, 18.5],
        [10.5, 11.5, 12.5, 13.5],
        [5.5,  6.5,  7.5,  8.5],
        [0.5,  1.5,  2.5,  3.5],
        ])
    actual = interpolate(state, direction='x')
    assert mock == pytest.approx(actual.matrix)


def test_interpolate_y():
    """Tests that the interpolation works in the y-direction."""
    Nx, Ny = 4, 5
    state = State(np.arange(Nx*Ny), Nx=Nx, Ny=Ny)
    mock = np.array([
        [14., 15., 16., 17.],
        [10., 11., 12., 13.],
        [6.,  7.,  8.,  9.],
        [2.,  3.,  4.,  5.],
        ])
    actual = interpolate(state, direction='y')
    assert mock == pytest.approx(actual.matrix)

# # Testing get_vector <-> get_matrix
# Nx, Ny = 5, 4
# matrix = np.flip(np.arange(1, Nx*Ny+1).reshape((Ny, Nx)), axis=0)
# print(matrix)
# vector = states.get_vector(matrix)
# print(vector)
# matrix2 = states.get_matrix(vector, Nx, Ny)
# print(matrix2)

# # convert to vector
# b = states.get_vector(b)
# print('b to vector')
# print(b)
# bmatrix = states.get_matrix(b, 4, 5)
# print('b vector to matrix')
# print(bmatrix)
# b2 = states.get_vector(bmatrix)
# print('b back to vector')
# print(b2)