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
