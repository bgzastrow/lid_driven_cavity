"""Contains Operator class."""
from dataclasses import dataclass, field
from typing import List
import numpy as np
import scipy as sp

from lid_driven_cavity import states


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
    row: List[int] = field(default_factory=list)
    col: List[int] = field(default_factory=list)
    data: List[float] = field(default_factory=list)

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


def compute_gradient_x(state):
    """
    Assembles a backward-differencing gradient operator in the x direction.
    """
    h = 1 / (state.Nx-1)
    G = sp.sparse.diags(
        diagonals=[-1.0/h, 1.0/h],
        offsets=[-1, 0],
        shape=(state.Nx*state.Ny, state.Nx*state.Ny),
        format="csr",
    )
    gradient = G.dot(state.vector)
    return states.State(gradient, Nx=state.Nx, Ny=state.Ny)


def compute_gradient_y(state):
    """
    Assembles a backward-differencing gradient operator in the y direction.
    """
    h = 1 / (state.Ny-1)
    G = sp.sparse.diags(
        diagonals=[-1.0/h, 1.0/h],
        offsets=[-state.Nx, 0],
        shape=(state.Nx*state.Ny, state.Nx*state.Ny),
        format="csr",
    )
    gradient = G.dot(state.vector)
    return states.State(gradient, Nx=state.Nx, Ny=state.Ny)


def assemble_laplacian_operator_u(b, Nx, Ny, h, k):
    """
    Computes the discrete laplacian (viscous operator).

    Parameters
    ----------
    q : (N^2,)
        State vector
    h : float
        spatial grid spacing (1/N)

    Returns
    -------
    laplacian : State
    """
    n = Nx*Ny
    operator = Operator((n, n))
    hm2 = h**(-2)

    # loop on internal grid points for now
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            # pointers
            row = lij(i, j, Nx)    # this is the row
            ip1 = lij(i+1, j, Nx)  # (i+1,j)
            im1 = lij(i-1, j, Nx)  # (i-1,j)
            jp1 = lij(i, j+1, Nx)  # (i,j+1)
            jm1 = lij(i, j-1, Nx)  # (i,j-1)

            # add coefficients
            operator.append(row, row, -4*hm2)  # diagonal
            operator.append(row, ip1, hm2)
            operator.append(row, im1, hm2)
            operator.append(row, jp1, hm2)
            operator.append(row, jm1, hm2)

    # top and bottom surfaces
    jtop = Ny-1
    jbottom = 0
    for i in range(1, Nx-1):

        # top surface (y=1)
        row = lij(i, jtop, Nx)  # this is the row
        jm1 = lij(i, jtop-1, Nx)  # (i,j-1)
        operator.append(row, row, 1)
        operator.append(row, jm1, 1)
        b[row] = 2*h  # TODO add gradient terms

        # bottom surface (y=0)
        row = lij(i, jbottom, Nx)  # this is the row
        jp1 = lij(i, jbottom+1, Nx)  # (i,j-1)
        operator.append(row, row, 1)
        operator.append(row, jp1, 1)
        b[row] = 0  # TODO add gradient terms

    # left and right surfaces
    iright = Nx-1
    ileft = 0
    for j in range(1, Ny-1):

        # right surface (x=1)
        row = lij(iright, j, Nx)  # this is the row
        operator.append(row, row, 1)
        b[row] = 0  # TODO add gradient terms

        # left surface (x=0)
        row = lij(ileft, j, Nx)  # this is the row
        operator.append(row, row, 1)
        b[row] = 0  # TODO add gradient terms

    return operator, b


def assemble_laplacian_operator_v(b, Nx, Ny, h, k):
    """
    Computes the discrete laplacian (viscous operator).

    Parameters
    ----------
    q : (N^2,)
        State vector
    h : float
        spatial grid spacing (1/N)

    Returns
    -------
    laplacian : State
    """
    n = Nx*Ny
    operator = Operator((n, n))
    hm2 = h**(-2)

    # loop on internal grid points for now
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            # pointers
            row = lij(i, j, Nx)    # this is the row
            ip1 = lij(i+1, j, Nx)  # (i+1,j)
            im1 = lij(i-1, j, Nx)  # (i-1,j)
            jp1 = lij(i, j+1, Nx)  # (i,j+1)
            jm1 = lij(i, j-1, Nx)  # (i,j-1)

            # add coefficients
            operator.append(row, row, -4*hm2)  # diagonal
            operator.append(row, ip1, hm2)
            operator.append(row, im1, hm2)
            operator.append(row, jp1, hm2)
            operator.append(row, jm1, hm2)

    # top and bottom surfaces
    jtop = Ny-1
    jbottom = 0
    for i in range(1, Nx-1):

        # top surface (y=1)
        row = lij(i, jtop, Nx)  # this is the row
        # jm1 = lij(i, jtop-1, Nx)  # (i,j-1)
        operator.append(row, row, 1)
        # operator.append(row, jm1, 1)
        b[row] = 0  # TODO add gradient terms

        # bottom surface (y=0)
        row = lij(i, jbottom, Nx)  # this is the row
        # jp1 = lij(i, jbottom+1, Nx)  # (i,j-1)
        operator.append(row, row, 1)
        # operator.append(row, jp1, 1)
        b[row] = 0  # TODO add gradient terms

    # left and right surfaces
    iright = Nx-1
    ileft = 0
    for j in range(1, Ny-1):

        # right surface (x=1)
        row = lij(iright, j, Nx)  # this is the row
        im1 = lij(iright-1, j, Nx)
        operator.append(row, row, 1)
        operator.append(row, im1, 1)
        b[row] = 0  # TODO add gradient terms

        # left surface (x=0)
        row = lij(ileft, j, Nx)  # this is the row
        ip1 = lij(ileft+1, j, Nx)
        operator.append(row, row, 1)
        operator.append(row, ip1, 1)
        b[row] = 0  # TODO add gradient terms

    return operator, b


def assemble_laplacian_operator_phi(b, u_star, v_star, Nx, Ny, h, k):
    """
    Computes the discrete laplacian (viscous operator).

    Parameters
    ----------
    q : (N^2,)
        State vector
    h : float
        spatial grid spacing (1/N)

    Returns
    -------
    laplacian : State
    """
    n = Nx*Ny
    operator = Operator((n, n))
    hm2 = h**(-2)

    # loop on internal grid points for now
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            # pointers
            row = lij(i, j, Nx)    # this is the row
            ip1 = lij(i+1, j, Nx)  # (i+1,j)
            im1 = lij(i-1, j, Nx)  # (i-1,j)
            jp1 = lij(i, j+1, Nx)  # (i,j+1)
            jm1 = lij(i, j-1, Nx)  # (i,j-1)

            # add coefficients
            operator.append(row, row, -4*hm2)  # diagonal
            operator.append(row, ip1, hm2)
            operator.append(row, im1, hm2)
            operator.append(row, jp1, hm2)
            operator.append(row, jm1, hm2)

    # top and bottom surfaces
    jtop = Ny-1
    jbottom = 0
    for i in range(1, Nx-1):

        # top surface (y=1)
        row = lij(i, jtop, Nx)  # this is the row
        jm1 = lij(i, jtop-1, Nx)  # (i,j-1)
        operator.append(row, row, 1)
        operator.append(row, jm1, -1)
        b[row] = (-h/k)*v_star[i, jtop]  # TODO add gradient terms

        # bottom surface (y=0)
        row = lij(i, jbottom, Nx)  # this is the row
        jp1 = lij(i, jbottom+1, Nx)  # (i,j-1)
        operator.append(row, row, 1)
        operator.append(row, jp1, -1)
        b[row] = (-h/k)*v_star[i, jbottom]  # TODO add gradient terms

    # left and right surfaces
    iright = Nx-1
    ileft = 0
    for j in range(1, Ny-1):

        # right surface (x=1)
        row = lij(iright, j, Nx)  # this is the row
        im1 = lij(iright-1, j, Nx)
        operator.append(row, row, 1)
        operator.append(row, im1, -1)
        b[row] = (-h/k)*u_star[iright, j]  # TODO add gradient terms

        # left surface (x=0)
        row = lij(ileft, j, Nx)  # this is the row
        ip1 = lij(ileft+1, j, Nx)
        operator.append(row, row, 1)
        operator.append(row, ip1, -1)
        b[row] = (-h/k)*u_star[ileft, j]  # TODO add gradient terms

    return operator, b

