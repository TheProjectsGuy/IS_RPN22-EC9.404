# Generates bernstein path through a single waypoint
"""
    Given a starting, ending point and a 'via' point (waypoint),
    generate a path passing through the points using Bezier curves.

    For this script, $n = 5$ (default value)
"""

# %% Import everything
import numpy as np
import os
from src.three_point_traj_planner import NonHoloThreePtBernstein
# Matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches as patch


# %% Functions
# Save figures
def gen_figs(x_vals, y_vals, start_pt, way_pt, end_pt,
        folder="./data/", rob_body_rad = 2.5, start_msize = 100.0, 
        waypt_msize = 100.0, end_msize = 100.0):
    r"""
        Save the trajectory figures in a folder.
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


# %%
if __name__ == "__main__":
    # ==== Begin: User configuration area ====
    # Points as [x, y]
    start_pt = [0, 0]
    end_pt = [40, 40]
    way_pt = [5, 30]
    # Time values
    to, tw, tf = [0, 20, 50]    # Start, waypoint, end
    # Other parameters
    ko, kw, kf = [0, np.tan(np.pi/5), 0]    # k = np.tan(theta)
    dko, dkw, dkf = [0, 0, 0]
    dxo, dxw, dxf = [0, 0, 0]
    # ==== End: User configuration area ====
    # %%
    # Convert to dictionary (for library)
    constraint_dict = {
        "to": to, "tw": tw, "tf": tf,
        "xo": start_pt[0], "xw": way_pt[0], "xf": end_pt[0],
        "yo": start_pt[1], "yw": way_pt[1], "yf": end_pt[1],
        "ko": ko, "kw": kw, "kf": kf,
        "dxo": dxo, "dxw": dxw, "dxf": dxf,
        "dko": dko, "dkw": dkw, "dkf": dkf
    }
    # Initialize solver
    path_solver = NonHoloThreePtBernstein()
    x_vals, y_vals, th_vals, t_vals, x_t, y_t, th_t = \
        path_solver.solve_wpt_constr(constraint_dict)
    # %%
    # Plot everything
    plt.figure(figsize=(5, 10))
    plt.subplot(3, 1, 1)
    plt.title("X plot")
    plt.plot(t_vals, x_vals, 'r')
    plt.subplot(3, 1, 2)
    plt.title("Y plot")
    plt.plot(t_vals, y_vals, 'g')
    plt.subplot(3, 1, 3)
    plt.title("Theta plot")
    plt.plot(t_vals, th_vals, 'b')
    plt.tight_layout()
    plt.show(block=False)
    # X Y plot
    plt.figure()
    plt.title("XY plot")
    plt.scatter(x_vals, y_vals, 1.0, c=t_vals)
    plt.colorbar()
    plt.show()
    # %%
    # Save the files
    gen_figs(x_vals, y_vals, start_pt, way_pt, end_pt)

# %% Experimental section

# %%
