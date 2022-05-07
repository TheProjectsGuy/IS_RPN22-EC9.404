# Drawing utilities for the program
"""
    Utilities to save visualization to disk
"""

# %%
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patch

# %%
# Save figures
def gen_wpt_figs(x_vals, y_vals, th_vals, start_pt, way_pt, end_pt,
        folder="./data/", rob_body_rad = 2.5, start_msize = 100.0, 
        waypt_msize = 100.0, end_msize = 100.0):
    r"""
        Save the trajectory figures in a folder. For waypoint and 
        bernstein polynomials (from original code).
    """
    # Folder
    folder = os.path.realpath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    # Get number of samples
    ns = len(x_vals)
    for i in range(ns):
        fig = plt.figure(dpi=200) # figsize=(7, 7), 
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal', 'datalim')
        ax.scatter(*start_pt, start_msize, 'c', 'o', label="Start", 
            zorder=4)
        ax.scatter(*way_pt, waypt_msize, 'g', '*', label="Waypoint", 
            zorder=4)
        ax.scatter(*end_pt, end_msize, 'r', '*', label="End", 
            zorder=4)
        # Plot the path (top-most)
        ax.plot(x_vals[:i], y_vals[:i], '-', zorder=5, label="Path")
        # Show the robot in the path
        rob_body = patch.Circle((x_vals[i], y_vals[i]), 
            radius=rob_body_rad, zorder=4.5, ec="black", 
            linewidth=1.0, ls='solid', alpha=0.75, fc="#88b4e6")
        ax.add_patch(rob_body)
        ax.plot(
            [x_vals[i], x_vals[i]+rob_body_rad*np.cos(th_vals[i])],
            [y_vals[i], y_vals[i]+rob_body_rad*np.sin(th_vals[i])],
            c='#7a0c7a', zorder=4.5, alpha=0.75)
        ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", 
            ncol=4, mode="expand")
        ax.set_xlim(np.min(x_vals) - 10, np.max(x_vals) + 10)
        ax.set_ylim(np.min(y_vals) - 10, np.max(y_vals) + 10)
        ax.grid(zorder=2.5)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.set_tight_layout(True)
        # plt.show(fig)
        fig.savefig(f"{folder}/{i}.png")
        fig.clear()
        plt.close(fig)
