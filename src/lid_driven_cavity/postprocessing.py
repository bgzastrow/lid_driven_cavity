"""
Tools for calculating and plotting performance, results, and error metrics.
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def relative_frobenius_error(x_numerical, x_analytical):
    """
    Computes the relative error in the frobenius norm
    """
    absolute_error = sp.linalg.norm(x_analytical-x_numerical)
    relative_error = absolute_error / sp.linalg.norm(x_analytical, ord="fro")
    return relative_error


def plot_convergence(hs, errors, reference_rates, offset=1.0):
    """
    Plots log-log convergence rate of error.
    Parameters
    ----------
    hs: array_like
        list of grid spacing values
    errors: array_like
        list of error values (in this case, absolute difference)
    reference_rates: array_like
        list of integers for reference rates to plot
    """

    fig, ax = plt.subplots()

    # Plot reference convergence rate lines
    xx = offset*np.array([
        1.0/hs[0],
        1.0/hs[-1],
        ])
    for reference_rate in reference_rates:
        yy = errors[0]*np.array([
            1.0,
            (xx[-1]/xx[0])**(-1*reference_rate),
            ])
        ax.loglog(
            xx,
            yy,
            color='black',
            alpha=0.3,
            linestyle='--',
            label=f'$O(h^{reference_rate:d})$',
            )

    # Plot errors vs. resolution (grid spacing)
    ax.loglog(
        1/np.array(hs),
        errors,
        marker='o',
        label='error',
        )

    # Annotate lines (instead of using legend)
    ax.spines[['right', 'top']].set_visible(False)
    for line in ax.lines[:-1]:
        x, y = line.get_data()
        xright, yright = x[-1], y[-1]
        ax.annotate(
            line.get_label(),
            xy=(xright, yright),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=line.get_color(),
        )
    ax.legend().set_visible(False)
    ax.set_xlabel("1/h")
    ax.set_ylabel("absolute error")
    plt.show()

def plot_streamlines(u, v, phi):
    N = phi.get_matrix().shape[0] - 2
    # u grid
    Ny, Nx = u.get_matrix().shape
    xu = np.linspace(0.0, 1.0, Nx)
    yu = np.linspace(0.0, 1.0, Ny)
    Xu, Yu = np.meshgrid(xu, yu)
    print(Xu)
    print(Yu)

    # v grid
    Ny, Nx = v.get_matrix().shape
    xv = np.linspace(0.0, 1.0, Nx)
    yv = np.linspace(0.0, 1.0, Ny)
    Xv, Yv = np.meshgrid(xv, yv)

    # Phi grid
    Nx, Ny = phi.get_matrix().shape
    xphi = np.linspace(0.0, 1.0, Nx)
    yphi = np.linspace(0.0, 1.0, Ny)
    Xphi, Yphi = np.meshgrid(xphi, yphi)

    # Target (plotting) grid
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    XN, YN = np.meshgrid(x, y)

    # Interpolation
    phiN = sp.interpolate.griddata((Xphi.flatten(), Yphi.flatten()), phi.vector, (XN, YN))
    vN = sp.interpolate.griddata((Xv.flatten(), Yv.flatten()), v.vector, (XN, YN))
    uN = sp.interpolate.griddata((Xu.flatten(), Yu.flatten()), u.vector, (XN, YN))

    # Plotting
    fig, ax = plt.subplots()
    # plt.contourf(XN, YN, phiN, np.arange(-0.003, 0.003, 0.0005))
    plt.contourf(XN, YN, phiN)
    plt.colorbar()
    # plt.quiver(XN, YN, uN, vN, color='black', scale=0.005)
    plt.quiver(XN, YN, uN, vN, color='black')
    # plt.streamplot(XN, YN, uN, vN, color='black', linewidth=1.0)
    plt.show()