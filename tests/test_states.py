"""Tests the State class functionality."""
import pytest
import numpy as np

from lid_driven_cavity import states

NX, NY = 4, 5  # number of grid points in x, y directions


@pytest.fixture(name='Nx')
def create_mock_Nx():
    """Mock Nx, i.e. number of grid points in x direction."""
    return NX


@pytest.fixture(name='Ny')
def create_mock_Ny():
    """Mock Ny, i.e. number of grid points in y direction."""
    return NY


@pytest.fixture(name='vector')
def create_mock_vector():
    """Mock state vector."""
    vector = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    return vector


@pytest.fixture(name='matrix')
def create_mock_matrix():
    """Mock state matrix."""
    matrix = np.array([
        [16, 17, 18, 19],
        [12, 13, 14, 15],
        [ 8,  9, 10, 11],
        [ 4,  5,  6,  7],
        [ 0,  1,  2,  3],
    ])
    return matrix


@pytest.fixture(name='matrix_padded')
def create_mock_padded_matrix():
    """Mock state matrix."""
    matrix = np.array([
        [0,  0,  0,  0,  0, 0],
        [0, 16, 17, 18, 19, 0],
        [0, 12, 13, 14, 15, 0],
        [0,  8,  9, 10, 11, 0],
        [0,  4,  5,  6,  7, 0],
        [0,  0,  1,  2,  3, 0],
        [0,  0,   0,  0,  0, 0],
    ])
    return matrix


@pytest.fixture(name='i', params=range(NX))
def create_mock_i(request):
    """Mock int value(s)."""
    return request.param


@pytest.fixture(name='j', params=range(NY))
def create_mock_j(request):
    """Mock int value(s)."""
    return request.param


def test_matrix(Nx, Ny, vector, matrix):
    """Tests that the correct matrix is created from the vector."""
    state = states.State(vector, Nx=Nx, Ny=Ny)
    assert matrix == pytest.approx(state.get_matrix())


def test_vector(Nx, Ny, vector, matrix):
    """Tests that the correct matrix is created from the vector."""
    state = states.State(matrix, Nx=Nx, Ny=Ny)
    assert vector == pytest.approx(state.vector)


def test_ij(i, j, matrix, Nx, Ny):
    """Tests that ij indexing function matches matrix output."""
    state = states.State(matrix, Nx=Nx, Ny=Ny)
    assert state.ij(i, j) == pytest.approx(matrix[Ny-j-1, i])


def test_vector_changes(Nx, Ny, vector, matrix):
    """Tests that updates to State.vector propogate correctly."""
    state = states.State(vector, Nx=Nx, Ny=Ny)
    state.vector[:Nx] = 100
    matrix[-1, :] = 100
    assert matrix == pytest.approx(state.get_matrix())


def test_matrix_changes(Nx, Ny, vector, matrix):
    """Tests that updates to State.matrix propogate correctly."""
    state = states.State(matrix, Nx=Nx, Ny=Ny)
    matrix = state.get_matrix()
    matrix[-1, :] = 100
    state.set_matrix(matrix)
    vector[:Nx] = 100
    assert vector == pytest.approx(state.vector)


def test_pad_boundaries(matrix, matrix_padded):
    """Tests pad_boundaries() method."""
    assert matrix_padded == pytest.approx(states.pad_boundaries(matrix))


def test_strip_boundaries(matrix, matrix_padded):
    """Tests strip_boundaries() method."""
    assert matrix == pytest.approx(states.strip_boundaries(matrix_padded))
