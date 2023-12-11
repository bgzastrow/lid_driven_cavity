"""Tests the Operator class functionality."""
import pytest
import numpy as np

from lid_driven_cavity.operators import Operator


@pytest.fixture(name='mock_vector')
def create_mock_vector():
    """Mock numpy vector."""
    mock = np.array([2, 5, -3, 4])
    return mock


@pytest.fixture(name='mock_array')
def create_mock_array():
    """Mock numpy array."""
    mock = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.14, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.48],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, -2.4, 0.0, 0.0],
    ])
    return mock


@pytest.fixture(name='mock_operator')
def create_mock_operator():
    """Mock Operator object"""
    mock = Operator(
        shape=(5, 4),
        row=[1, 2, 4, 2],
        col=[0, 3, 1, 2],
        data=[0.14, 0.48, -2.4, 10.0],
        )
    return mock


def test_get_csr_1(mock_operator, mock_array):
    """Tests successful construction of CSR matrix and reconversion to dense
    matrix."""
    assert mock_array == pytest.approx(mock_operator.get_csr().toarray())


def test_get_csr_2(mock_operator, mock_array):
    """Tests failed construction of CSR matrix and reconversion to dense
    matrix."""
    mock_array_wrong = mock_array*2.0
    assert mock_array_wrong != pytest.approx(mock_operator.get_csr().toarray())


def test_multiply_1(mock_operator, mock_array, mock_vector):
    """Tests matrix-vector product of sparse operator with a vector."""
    actual = mock_array @ mock_vector
    assert actual == pytest.approx(mock_operator.multiply(mock_vector))


def test_multiply_2(mock_operator, mock_vector):
    """Tests matrix-vector product of sparse operator with a vector more
    nicely."""
    mock_array = mock_operator.get_csr().toarray()
    actual = mock_array @ mock_vector
    assert actual == pytest.approx(mock_operator.multiply(mock_vector))

# # make dummy b matrix for debugging
# vector = np.arange(1,b.shape[0]*b.shape[1]+1)
# vector2 = vector.reshape(b.shape)
# b = np.flip(vector2, axis=0)
# print('b')
# print(b)
