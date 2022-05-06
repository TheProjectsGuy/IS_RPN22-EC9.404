# Visualize output cache
"""
"""

# %% Import everything
import numpy as np
import joblib
from rrt_nonh_dd_solver import RRTSolverDiffDrive
from rrt_nonh_dd_solver import Map, Rectangle, Circle, Obstacle
from matplotlib import pyplot as plt
import os

# %%

# %% Experimental

# %%
def save_path_figs(xyth_path, vw_comm, rrt_solver, start_pose, 
        end_pose, out_dir="./out_path"):
    if out_dir is not None:
        out_dir = os.path.realpath(os.path.expanduser(out_dir))
        # Check output directory
        if not os.path.isdir(out_dir):
            # Make directory
            print(f"[INFO]: Output directory '{out_dir}' created")
            os.makedirs(out_dir)
        else:
            print(f"[ERROR]: Output directory exists")
            raise FileExistsError(f"Output directory: {out_dir}")
    # For each element in path
    for i in range(len(xyth_path)):
        # Show map
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = fig.add_subplot()
        ax.set_aspect('equal', 'box')
        # Main map
        rrt_solver.draw_map_on_axes(ax, color='r')
        # Show the start and end point
        sx, sy, _ = start_pose
        ex, ey, _ = end_pose
        ax.plot(sx, sy, "b*", ms=10.0, zorder=1)
        ax.plot(ex, ey, "g*", ms=10.0, zorder=1)
        # Obstacles in the map (in red)
        rrt_solver.map.draw_on_axes(ax, "-.", color='r')
        # Start and end star markers
        ax.plot(sx, sy, "b*", ms=10.0)
        ax.plot(ex, ey, "g*", ms=10.0)
        # Draw tree
        for node in rrt_solver.node_tree:
            # Nodes (in grey)
            ax.plot(node.x, node.y, 'o', c='grey', ms=2.0)
            # Child connections
            for c in node.children: # 'c' is index
                xp, yp = node.x, node.y
                # Get five intermediates and join
                for t in np.linspace(0, 1.0, 20):  # Finer res.
                    xi, yi, _ = rrt_solver.fk_dd(node.x, node.y, 
                        node.theta, rrt_solver.node_tree[c].v, 
                        rrt_solver.node_tree[c].w, t)
                    # Draw it with lines
                    plt.plot([xi, xp], [yi, yp], '--', c='grey')
                    # Remember this point
                    xp, yp = xi, yi
        # Show the path progressing
        ax.plot(xyth_path[0:i, 0], xyth_path[0:i, 1], 'co')
        # Show all controls from 0 through i
        for ci in range(i):
            xp, yp = xyth_path[ci, 0], xyth_path[ci, 1]
            for t in np.linspace(0, 1.0, 30):   # Very fine!
                xi, yi, _ = rrt_solver.fk_dd(xyth_path[ci, 0], 
                    xyth_path[ci, 1], xyth_path[ci, 2], 
                    vw_comm[ci, 0], vw_comm[ci, 1], t)
                # Draw curve
                plt.plot([xi, xp], [yi, yp], 'c--')
                xp, yp = xi, yi
        # fig.set_tight_layout(True)
        # Save figure
        if out_dir is not None:
            fig.savefig(f"{out_dir}/{i}.png")
        else:
            try:
                plt.show(fig)
            except:
                plt.show()
        # Clear the figure and buffer
        fig.clear()
        plt.close('all')

# %%
# Show the robot moving on the path
def display_rob_traj(start_pose, end_pose, xyth_path, vw_comm, 
        rrt_solver, b=1., out_dir="./out_wheels"):
    """
        Show a differential drive robot with baseline 'b' travelling
        on the given path.
    """
    def _get_lrwpos(center_pt):
        # Given center_pt: (x, y, th), get left & right wheel pose
        cx, cy, cth = center_pt
        tf_mat = np.array([
            [np.cos(cth), -np.sin(cth), cx],
            [np.sin(cth), np.cos(cth), cy],
            [0, 0, 1]
        ])
        lw = (tf_mat @ np.array([0, b/2, 1]).reshape(-1, 1)).\
            flatten()
        lw_pose = [lw[0], lw[1], cth]    # Pose of the left wheel
        rw = (tf_mat @ np.array([0, -b/2, 1]).reshape(-1, 1)).\
            flatten()
        rw_pose = [rw[0], rw[1], cth]    # Pose of the left wheel
        return lw_pose, rw_pose
    if out_dir is not None:
        out_dir = os.path.realpath(os.path.expanduser(out_dir))
        # Check output directory
        if not os.path.isdir(out_dir):
            # Make directory
            print(f"[INFO]: Output directory '{out_dir}' created")
            os.makedirs(out_dir)
        else:
            print(f"[ERROR]: Output directory exists")
            raise FileExistsError(f"Output directory: {out_dir}")
    # Calculate the points in the path using xyth, vw
    rc_path = []    # Center
    rl_path = []    # Left wheel
    rr_path = []    # Right wheel
    for i in range(len(xyth_path)):
        # x, y
        rc_path.append([xyth_path[i, 0], xyth_path[i, 1], 
            xyth_path[i, 2]])
        lpose, rpose = _get_lrwpos(xyth_path[i])
        rl_path.append(lpose)
        rr_path.append(rpose)
        # Append control points
        if i < len(xyth_path) - 1:
            # Apply control[i]
            v, w = vw_comm[i]
            for t in np.linspace(0, 1.0, 30):   # Very fine!
                xi, yi, thi = rrt_solver.fk_dd(xyth_path[i,0], 
                    xyth_path[i,1], xyth_path[i,2], v, w, t)
                rc_path.append([xi, yi, thi])
                lpose, rpose = _get_lrwpos([xi, yi, thi])
                rl_path.append(lpose)
                rr_path.append(rpose)
    # Convert to numpy
    rc_path = np.array(rc_path)
    rl_path = np.array(rl_path)
    rr_path = np.array(rr_path)
    for i in range(len(rc_path)):
        # Draw the map
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = fig.add_subplot()
        ax.set_aspect('equal', 'box')
        # Main map
        rrt_solver.draw_map_on_axes(ax, color='r')
        # Show the start and end point
        sx, sy, _ = start_pose
        ex, ey, _ = end_pose
        ax.plot(sx, sy, "b*", ms=10.0, zorder=1)
        ax.plot(ex, ey, "g*", ms=10.0, zorder=1)
        # Obstacles in the map (in red)
        rrt_solver.map.draw_on_axes(ax, "-.", color='r')
        # Start and end star markers
        ax.plot(sx, sy, "b*", ms=10.0)
        ax.plot(ex, ey, "g*", ms=10.0)
        # Draw path from 0 to 'i' (in path)
        ax.plot(rc_path[:i, 0], rc_path[:i, 1], 'k-')
        ax.plot(rl_path[:i, 0], rl_path[:i, 1], 'r--', label="L")
        ax.plot(rr_path[:i, 0], rr_path[:i, 1], 'b--', label="R")
        # Legend
        ax.legend()
        # Save fig
        fig.savefig(f"{out_dir}/{i}.png")
        # Clear everything
        fig.clear()
        plt.close('all')

# %% Load file
cache_out = "./output_cache"
res = joblib.load(cache_out)    # Results

# %%
# Recover poses
start_pose = res["start_pose"]
end_pose = res["end_pose"]
rrt_solver: RRTSolverDiffDrive = res["solver"]
xyth_path, vw_comm = res["result"]

# %% Show the path evolution
save_path_figs(xyth_path, vw_comm, rrt_solver, start_pose, end_pose)

# %%
rc_path = display_rob_traj(start_pose, end_pose, xyth_path, vw_comm, 
    rrt_solver)

# %%
