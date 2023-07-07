from pathlib import Path

import fenics as fe
import matplotlib.pyplot as plt
import numpy as np


def plot_optimisation_convergence(j_values: np.ndarray, outpath: Path):
    """Line plot of progress of optimisation
    """
    plt.figure()
    plt.plot(j_values)
    plt.title('Error functional value')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.savefig(outpath)



def plot_fe_function_comparison(f1: fe.Function, f2: fe.Function, label1: str, label2: str, filepath: Path = None):
    """Plot comparison of two Fenics functions
    """
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    axs = np.ravel(axs)

    plt.sca(axs[0])
    c = fe.plot(f1, title=label1)
    fig.colorbar(c, ax=axs[0])


    plt.sca(axs[1])
    c = fe.plot(f2, title=label2)
    fig.colorbar(c, ax=axs[1])

    cmap = 'coolwarm'

    minpoint = fe.Point(f1.function_space().mesh().org_mesh_coords[0])
    maxpoint = fe.Point(f1.function_space().mesh().org_mesh_coords[-1])
    xs = np.linspace(minpoint[0], maxpoint[0], 100)
    ys = np.linspace(minpoint[1], maxpoint[1], 100)
    rs = [np.sqrt(x**2 + y**2) for x, y in zip(xs, ys)]
    f1_vals = [f1(x, y) for x, y in zip(xs, ys)]
    f2_vals = [f2(x, y) for x, y in zip(xs, ys)]

    axs[2].set_title('Main diagonal 1D lineout')
    axs[2].plot(rs, f1_vals, label=label1)
    axs[2].plot(rs, f2_vals, label=label2)
    axs[2].legend()

    plt.sca(axs[3])
    im = fe.plot(f1 - f2, title='Difference', cmap=cmap)
    fig.colorbar(im, ax=axs[3])
    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()


