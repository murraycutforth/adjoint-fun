import math
from pathlib import Path
import logging

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def plot_final(u, f, a, filename):
    """Plot source term, velocity field, and final temperature solution
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    plt.clf()
    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    c = fe.plot(u)
    cbar = fig.colorbar(c, ax=ax1)
    cbar.set_label("[temperature]")
    ax1.set_title("Solved temperature")

    ax2 = fig.add_subplot(2, 2, 2)
    c = fe.plot(f)
    cbar = fig.colorbar(c, ax=ax2)
    cbar.set_label("[temperature / time]")
    ax2.set_title("Source term")

    ax3 = fig.add_subplot(2, 2, 3)
    c = fe.plot(a)
    cbar = fig.colorbar(c, ax=ax3)
    cbar.set_label("[velocity]")
    ax3.set_title("Velocity field")

    fig.tight_layout()
    plt.savefig(filename)
    logger.info(f"Written plot to: {filename}")


def plot_final_param_fields(f, a, filename):
    """Plot source term, velocity field
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    plt.clf()
    fig = plt.figure(figsize=(12, 6))

    ax2 = fig.add_subplot(1, 2, 1)
    c = fe.plot(f)
    cbar = fig.colorbar(c, ax=ax2)
    cbar.set_label("[temperature / time]")
    ax2.set_title("Source term")

    ax3 = fig.add_subplot(1, 2, 2)
    c = fe.plot(a)
    cbar = fig.colorbar(c, ax=ax3)
    cbar.set_label("[velocity]")
    ax3.set_title("Velocity field")

    fig.tight_layout()
    plt.savefig(filename)
    logger.info(f"Written plot to: {filename}")


def plot_timestep_solns(V, u_ts_filename, T, num_images, filename, separate_plots = False):
    """Plot temperature solution at many time points
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    plt.clf()
    fig = plt.figure(figsize=(12, 12))

    n_side = math.ceil(math.sqrt(num_images))
    u_ts = fe.TimeSeries(u_ts_filename)
    u = fe.Function(V)

    for k, t in enumerate(np.linspace(0.0, T, num_images)):
        if separate_plots:
            plt.clf()
            ax = plt.gca()
        else:
            ax = fig.add_subplot(n_side, n_side, k + 1)

        u_ts.retrieve(u.vector(), t)

        c = fe.plot(u, vmin=0.0, vmax=0.09)
        cbar = fig.colorbar(c, ax=ax)
        ax.set_title(f"time={t:.2f}")

        if separate_plots:
            fig.tight_layout()
            plt.savefig(filename[:-4] + f"_{k}.png")
            logger.info(f"Written plot to: {filename}")

    if separate_plots:
        pass
    else:
        fig.tight_layout()
        plt.savefig(filename)
        logger.info(f"Written plot to: {filename}")


