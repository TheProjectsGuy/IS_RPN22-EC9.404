# Testing differential drive kinematics
"""
"""

# %%
import numpy as np
import sympy as sp

# %%
# Variables
x, y, th = sp.symbols(r"x, y, \theta")
v, w = sp.symbols(r"v, \omega")
dt = sp.symbols(r"dt")

# %%
# Radius of curvature
R = v/w
# Current instantaneous center of curvature
curr_icc = sp.Matrix([x-R*sp.sin(th), y+R*sp.cos(th)])
rot_mat = sp.Matrix([   # Rotation matrix for R vector
    [sp.cos(w*dt), -sp.sin(w*dt)],
    [sp.sin(w*dt),  sp.cos(w*dt)]
])

# %%
curr_pos = sp.Matrix([x, y])
next_pos = rot_mat @ (curr_pos - curr_icc) + curr_icc
curr_ang = th
next_ang = th + w*dt
# Pose = [x, y, theta]
next_pose = sp.Matrix([next_pos[0], next_pos[1], next_ang])
next_pose = sp.simplify(next_pose)

# %% Forward kinematics definition
def fk_dd(cx=0., cy=0., cth=0., vc=0., wc=0., dt=0.1):
    r"""
        Forward kinematics of a (2-wheeled) differential drive robot.
        Defaults for all paramters is 0.0, other than `dt` which is
        0.2.

        Parameters:
        - cx, cy, cth: Current pose (x, y, theta) 
        - vc, wc: Linear and angular velocity commands 
        - dt: Change in time (for next step)

        Returns:
        - nx, ny, nth: Next frame pose (x, y, theta)
    """
    min_odom = 1e-3
    nx, ny = 0., 0. # Next pose
    # Straight line or curved line
    if abs(wc) < min_odom:  # Straight line
        nx = cx + vc*np.cos(cth)*dt
        ny = cy + vc*np.sin(cth)*dt
    else:   # Curved path
        nx = cx + (-vc*np.sin(cth) + vc*np.sin(wc*dt + cth))/wc
        ny = cy + (vc*np.cos(cth) - vc*np.cos(wc*dt + cth))/wc
    nth = cth + wc*dt
    return nx, ny, nth

# %%
x_val, y_val, th_val = 0., 0., np.deg2rad(0.)
v_val, w_val = 0.2, 0.1

# %%
