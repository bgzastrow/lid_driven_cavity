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


def compute_interpolated_x(state):
    """Interpolated (average) state in x direction (to midpoints)."""
    interpolated_matrix = (state.get_matrix()[:, 1:] + state.get_matrix()[:, :-1]) / 2.0
    return states.State(interpolated_matrix, Nx=state.Nx-1, Ny=state.Ny)


def compute_interpolated_y(state):
    """Interpolated (average) state in y direction (to midpoints)."""
    interpolated_matrix = (state.get_matrix()[1:, :] + state.get_matrix()[:-1, :]) / 2.0
    return states.State(interpolated_matrix, Nx=state.Nx, Ny=state.Ny-1)


def compute_gradient_x_centered(state):
    """
    Assembles a central-difference gradient operator in the x direction.
    """
    h = 1 / (state.Nx-1)
    G = sp.sparse.diags(
        diagonals=[-1.0/(2*h), 1.0/(2*h)],
        offsets=[-1, 1],
        shape=(state.Nx*state.Ny, state.Nx*state.Ny),
        format="csr",
    )
    gradient = G.dot(state.vector)
    return states.State(gradient, Nx=state.Nx, Ny=state.Ny)


def compute_gradient_y_centered(state):
    """
    Assembles a central-difference gradient operator in the y direction.
    """
    h = 1 / (state.Ny-1)
    G = sp.sparse.diags(
        diagonals=[-1.0/(2*h), 1.0/(2*h)],
        offsets=[-state.Nx, state.Nx],
        shape=(state.Nx*state.Ny, state.Nx*state.Ny),
        format="csr",
    )
    gradient = G.dot(state.vector)
    return states.State(gradient, Nx=state.Nx, Ny=state.Ny)


def compute_Nu(u_n, v_hat_n, h):
    """Computes convective (nonlinear) term for u."""
    # Compute gradient of u
    dudx = compute_gradient_x_centered(u_n)  # on u grid, no boundary columns
    dudy = compute_gradient_y_centered(u_n)  # on u grid, no boundary rows
    dudx = dudx.get_matrix()[1:-1, 1:-1]
    dudy = dudy.get_matrix()[1:-1, 1:-1]

    # Get interior points of u_n
    u_n_interior = u_n.get_matrix()[1:-1, 1:-1]

    # Interpolate v_hat in x direction
    v_hat_interpolated = compute_interpolated_x(v_hat_n)  # on u grid, no boundary rows
    v_hat_interpolated = v_hat_interpolated.get_matrix()[:, 1:-1]

    # [u, v] dot grad(u)
    Nu_interior = (-u_n_interior*dudx - v_hat_interpolated*dudy) / (2*h)
    Nu = states.State(Nu_interior, Nx=Nu_interior.shape[1], Ny=Nu_interior.shape[0])
    Nu.pad_boundaries()

    return Nu


def compute_Nv(v_n, u_hat_n, h):
    """Computes convective (nonlinear) term for u."""
    # Compute gradient of v
    dvdx = compute_gradient_x_centered(v_n)  # on u grid, no boundary columns
    dvdy = compute_gradient_y_centered(v_n)  # on u grid, no boundary rows
    dvdx = dvdx.get_matrix()[1:-1, 1:-1]
    dvdy = dvdy.get_matrix()[1:-1, 1:-1]

    # Get interior points of v_n
    v_n_interior = v_n.get_matrix()[1:-1, 1:-1]

    # Interpolate v_hat in x direction
    u_hat_interpolated = compute_interpolated_y(u_hat_n)  # on u grid, no boundary rows
    u_hat_interpolated = u_hat_interpolated.get_matrix()[1:-1, :]

    # [u, v] dot grad(u)
    Nv_interior = (-v_n_interior*dvdx - u_hat_interpolated*dvdy) / (2*h)
    Nv = states.State(Nv_interior, Nx=Nv_interior.shape[1], Ny=Nv_interior.shape[0])
    Nv.pad_boundaries()

    return Nv


def compute_laplace(state):
    """Computes Laplacian of state via 2nd order centered finite difference."""
    h = 1 / (state.Nx-1)
    h2 = h**2
    L = sp.sparse.diags(
        diagonals=[1.0/h2, 1.0/h2, -4.0/h2, 1.0/h2, 1.0/h2],
        offsets=[-state.Nx, -1, 0, 1, state.Nx],
        shape=(state.Nx*state.Ny, state.Nx*state.Ny),
        format="csr",
    )
    laplacian = L.dot(state.vector)
    return states.State(laplacian, Nx=state.Nx, Ny=state.Ny)


def assemble_laplacian_operator_u(b, Nx, Ny, h):
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
            row = states.lij(i, j, Nx)    # this is the row
            ip1 = states.lij(i+1, j, Nx)  # (i+1,j)
            im1 = states.lij(i-1, j, Nx)  # (i-1,j)
            jp1 = states.lij(i, j+1, Nx)  # (i,j+1)
            jm1 = states.lij(i, j-1, Nx)  # (i,j-1)

            # add coefficients
            operator.append(row, row, -4*hm2)  # diagonal
            operator.append(row, ip1, hm2)
            operator.append(row, im1, hm2)
            operator.append(row, jp1, hm2)
            operator.append(row, jm1, hm2)

    # top and bottom surfaces
    jtop = Ny-1
    jbottom = 0
    for i in range(0, Nx):

        # top surface (y=1)
        row = states.lij(i, jtop, Nx)  # this is the row
        jm1 = states.lij(i, jtop-1, Nx)  # (i,j-1)
        operator.append(row, row, 1)
        operator.append(row, jm1, 1)
        b[row] = 2  # TODO add gradient terms

        # bottom surface (y=0)
        row = states.lij(i, jbottom, Nx)  # this is the row
        jp1 = states.lij(i, jbottom+1, Nx)  # (i,j-1)
        operator.append(row, row, 1)
        operator.append(row, jp1, 1)
        b[row] = 0  # TODO add gradient terms

    # left and right surfaces
    iright = Nx-1
    ileft = 0
    for j in range(1, Ny-1):

        # right surface (x=1)
        row = states.lij(iright, j, Nx)  # this is the row
        operator.append(row, row, 1)
        b[row] = 0  # TODO add gradient terms

        # left surface (x=0)
        row = states.lij(ileft, j, Nx)  # this is the row
        operator.append(row, row, 1)
        b[row] = 0  # TODO add gradient terms

    return operator, b


def assemble_laplacian_operator_v(b, Nx, Ny, h):
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
            row = states.lij(i, j, Nx)    # this is the row
            ip1 = states.lij(i+1, j, Nx)  # (i+1,j)
            im1 = states.lij(i-1, j, Nx)  # (i-1,j)
            jp1 = states.lij(i, j+1, Nx)  # (i,j+1)
            jm1 = states.lij(i, j-1, Nx)  # (i,j-1)

            # add coefficients
            operator.append(row, row, -4*hm2)  # diagonal
            operator.append(row, ip1, hm2)
            operator.append(row, im1, hm2)
            operator.append(row, jp1, hm2)
            operator.append(row, jm1, hm2)

    # top and bottom surfaces
    jtop = Ny-1
    jbottom = 0
    for i in range(0, Nx):

        # top surface (y=1)
        row = states.lij(i, jtop, Nx)  # this is the row
        operator.append(row, row, 1)
        b[row] = 0  # TODO add gradient terms

        # bottom surface (y=0)
        row = states.lij(i, jbottom, Nx)  # this is the row
        operator.append(row, row, 1)
        b[row] = 0  # TODO add gradient terms

    # left and right surfaces
    iright = Nx-1
    ileft = 0
    for j in range(1, Ny-1):

        # right surface (x=1)
        row = states.lij(iright, j, Nx)  # this is the row
        im1 = states.lij(iright-1, j, Nx)
        operator.append(row, row, 1)
        operator.append(row, im1, 1)
        b[row] = 0  # TODO add gradient terms

        # left surface (x=0)
        row = states.lij(ileft, j, Nx)  # this is the row
        ip1 = states.lij(ileft+1, j, Nx)
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
            row = states.lij(i, j, Nx)    # this is the row
            ip1 = states.lij(i+1, j, Nx)  # (i+1,j)
            im1 = states.lij(i-1, j, Nx)  # (i-1,j)
            jp1 = states.lij(i, j+1, Nx)  # (i,j+1)
            jm1 = states.lij(i, j-1, Nx)  # (i,j-1)

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
        row = states.lij(i, jtop, Nx)  # this is the row
        jm1 = states.lij(i, jtop-1, Nx)  # (i,j-1)
        operator.append(row, row, 1)
        operator.append(row, jm1, -1)
        # b[row] = (-h/k)*v_star[i, jtop]  # TODO add gradient terms
        b[row] = 0.0  # TODO add gradient terms

        # bottom surface (y=0)
        row = states.lij(i, jbottom, Nx)  # this is the row
        jp1 = states.lij(i, jbottom+1, Nx)  # (i,j-1)
        operator.append(row, row, 1)
        operator.append(row, jp1, -1)
        # b[row] = (-h/k)*v_star[i, jbottom]  # TODO add gradient terms
        b[row] = 0.0  # TODO add gradient terms

    # left and right surfaces
    iright = Nx-1
    ileft = 0
    for j in range(1, Ny-1):

        # right surface (x=1)
        row = states.lij(iright, j, Nx)  # this is the row
        im1 = states.lij(iright-1, j, Nx)
        operator.append(row, row, 1)
        operator.append(row, im1, -1)
        # b[row] = (-h/k)*u_star[iright, j]  # TODO add gradient terms
        b[row] = 0.0  # TODO add gradient terms

        # left surface (x=0)
        row = states.lij(ileft, j, Nx)  # this is the row
        ip1 = states.lij(ileft+1, j, Nx)
        operator.append(row, row, 1)
        operator.append(row, ip1, -1)
        # b[row] = (-h/k)*u_star[ileft, j]  # TODO add gradient terms
        b[row] = 0.0  # TODO add gradient terms

    # Set useless ghost points to zero
    operator.append(0, 0, 1.0)  # bottom left
    operator.append(Nx-1, Nx-1, 1.0)  # bottom right
    operator.append(n-Nx-1, n-Nx-1, 1.0)  # top left
    operator.append(n-1, n-1, 1.0)  # top right

    return operator, b
