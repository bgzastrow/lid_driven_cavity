"""Tests the State class functionality."""
import pytest
import numpy as np

from lid_driven_cavity.states import State


@pytest.fixture(name='mock_vector')
def create_mock_vector():
    """Mock numpy vector."""
    mock = np.arange(0, 16)
    return mock


@pytest.fixture(name='mock_matrix')
def create_mock_matrix():
    """Mock 2d numpy array."""
    mock = np.array([
        [12, 13, 14, 15],
        [8,   9, 10, 11],
        [4,   5,  6,  7],
        [0,   1,  2,  3],
    ])
    return mock


@pytest.fixture(name='mock_state')
def create_mock_state():
    """Mock State object."""
    return State(np.arange(0, 16))


@pytest.fixture(name='i', params=range(4))
def create_mock_int1(request):
    """Mock int value(s)."""
    return request.param


@pytest.fixture(name='j', params=range(4))
def create_mock_int2(request):
    """Mock int value(s)."""
    return request.param


def test_get_matrix(mock_state, mock_matrix):
    """Tests that the correct matrix is created from the vector."""
    assert mock_matrix == pytest.approx(mock_state.get_matrix())


def test_matrix_equals_ij(mock_state, i, j):
    """Tests that ij indexing function matches matrix output."""
    matrix = mock_state.get_matrix()
    assert mock_state.ij(i, j) == pytest.approx(matrix[mock_state.N-j-1, i])
