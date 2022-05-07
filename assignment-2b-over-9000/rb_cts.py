# Rule-based Constant time-scaling
"""
    Given a robot trajectory generated through bernstein polynomials
    (modified to return the position and velocities), and the
    trajectory of a holonomic obstacle (straight line equation), we
    alter the robot's velocities (when the robot is in the collision
    bounds).
    This script assumes the following:
    - The robot can independently control two variables (using which
        the bernstein model is created): 'x', and 'tan(theta)'. 
        Scaling will be applied to these two variables
    - There should only be one collision with the obstacle and robot.
        The paths should not have multiple intersections (only one)
    
"""

# %% Import everything
# Main imports
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib import patches as patch
# For trajectory generation
from lib.three_point_traj_planner import NonHoloThreePtBernstein
from lib.ct_scaling import sp_time_scaling_eq

# %%

# %%

# %% Experimental section
# Generate a random robot trajectory
# ==== Begin: User configuration area (robot trajectory) ====
# Points as [x, y]
start_pt = [0, 0]
end_pt = [40, 40]
way_pt = [20, 25]
# Time values
to, tw, tf = [0., 25., 50.]    # Start, waypoint, end
# Other parameters
ko, kw, kf = [0, np.tan(np.pi/4), 0]    # k = np.tan(theta)
dko, dkw, dkf = [0, 0, 0]
dxo, dxw, dxf = [0, 1, 0]
# ==== End: User configuration area (robot trajectory) ====
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
# Time symbol
t_sp = sp.symbols('t', real=True, positive=True)
t_all = sp.symbols('t')
# Solve for paths
x_vals, y_vals, th_vals, t_vals, x_t, y_t, th_t = \
    path_solver.solve_wpt_constr(constraint_dict)
# Substitute 't' with real and positive 't' (time substitution)
x_t = x_t.subs({t_all: t_sp})
y_t = y_t.subs({t_all: t_sp})
th_t = th_t.subs({t_all: t_sp})
# Plot trajectories
plt.figure()
plt.title("XY plot")
plt.scatter(x_vals, y_vals, 1.0, c=t_vals)
plt.colorbar()
plt.show()

# %% Collision with an obstacle
# ==== Begin: User configuration area (obstacle) ====
obs_t_col = 25      # Time of collision (for x, y intermediate)
obs_start = (5, 35)     # (x, y): Starting point of obstacle
# ==== End: User configuration area (obstacle) ====
ox_i = float(x_t.subs({t_sp: obs_t_col}))
oy_i = float(y_t.subs({t_sp: obs_t_col}))
obs_x_t = obs_start[0] + ((ox_i - obs_start[0])/obs_t_col) * t_sp
obs_y_t = obs_start[1] + ((oy_i - obs_start[1])/obs_t_col) * t_sp
# Time, x, y trajectories (array) - visualize
obs_t_vals = t_vals.copy()  # np.linspace(to, tf, 100)
obs_x_vals = np.array([obs_x_t.subs({t_sp: tv}) for tv in obs_t_vals])
obs_y_vals = np.array([obs_y_t.subs({t_sp: tv}) for tv in obs_t_vals])

# %% Show the collision
plt.figure()
plt.title("XY plot")
plt.scatter(x_vals, y_vals, 1.0, c=t_vals)
plt.scatter(obs_x_vals, obs_y_vals, 1.0, c=t_vals)
plt.colorbar()
plt.show()

# %%

# %% Show the evolution as time functions
obs_rad = 1     # Obstacle radius
rob_rad = 2     # Robot radius
# Show the figure
fig = plt.figure(num="Original Trajectory")
ax = fig.add_subplot()
ax.set_aspect('equal')
# v_i = 49
# if True:
for v_i in range(len(t_vals)):  # FIXME: Don't run in VSCode (15s!)
    # Reset animation
    ax.cla()
    # Show the obstacle
    obs_body = patch.Circle((obs_x_vals[v_i], obs_y_vals[v_i]), 
        obs_rad, ec='k', fc="#F06767", zorder=3.5)
    # Show the robot
    rob_body = patch.Circle((x_vals[v_i], y_vals[v_i]), rob_rad, 
        ec='k', fc="#88B4E6", alpha=0.5, zorder=3.4)
    # Add patches
    ax.add_patch(obs_body)
    ax.add_patch(rob_body)
    # Show the paths
    ax.plot(obs_x_vals, obs_y_vals, alpha=0.5, zorder=3)
    ax.plot(x_vals, y_vals, alpha=0.5, zorder=3)
    ax.plot(ox_i, oy_i, 'kx', zorder=3)
    # Set limits
    ax.set_xlim(start_pt[0]-5, end_pt[0]+5)
    ax.set_ylim(start_pt[1]-5, end_pt[1]+5)
    # Pause simulation
    plt.pause(0.1)

# %% Collision Avoidance
# ==== Begin: User configuration area (collision avoidance) ====
collav_dist = 5    # Sensor activation distance
k_val = 0.25         # Scaling to apply (to the speed)
# ==== End: User configuration area (collision avoidance) ====
# Distance as time goes
dist_t = ((x_t - obs_x_t)**2 + (y_t - obs_y_t)**2)**0.5 - rob_rad - \
    obs_rad
dist_vals = np.array([dist_t.subs({t_sp: tv}) for tv in t_vals], 
    float)
# Time of collision
t_si, t_ei = np.where(dist_vals < collav_dist)[0][[0, -1]]
t_cstart = t_vals[t_si]     # Time of start (for collision)
t_cend = t_vals[t_ei]       # Time of end of collision
# Plot the trajectory
plt.figure()
plt.plot(t_vals, dist_vals)
plt.axhline(collav_dist, color='r', ls='--')
plt.axvline(t_cstart, color='k', ls='--')
plt.axvline(t_cend, color='k', ls='--')
plt.show()

# %%
# - Main collision avoidance work (const. time scaling, rule based) -
nx_t, t2_new, tend_new = sp_time_scaling_eq(x_t, t_cstart, t_cend, 
    [to, tf], k_val)
ny_t, t2_new, tend_new = sp_time_scaling_eq(y_t, t_cstart, t_cend,
    [to, tf], k_val)
# Do this operation over `tan(theta)` instead of `theta`
k_t = sp.tan(th_t)
nk_t, t2_new, tend_new = sp_time_scaling_eq(k_t, t_cstart, t_cend,
    [to, tf], k_val)
nth_t = sp.atan(k_t)    # Retrieve new theta(t) - This WON'T work
# (Precisely because the system is non-holonomic)
# Backup angle (of path) - Reinforce the theta constraint
nth_t = sp.atan2(ny_t.diff(t_sp), nx_t.diff(t_sp))

# %% Change the original and the obstacle trajectory (for viz.)
# New obstacle trajectory (stay at rest in the end)
new_obs_x_t = sp.Piecewise((obs_x_t, t_sp < tf), 
    (obs_x_t.subs({t_sp: tf}), True))
new_obs_y_t = sp.Piecewise((obs_y_t, t_sp < tf), 
    (obs_y_t.subs({t_sp: tf}), True))
# New original robot trajectories (stay at rest in the end)
new_orig_x_t = sp.Piecewise((x_t, t_sp < tf), 
    (x_t.subs({t_sp: tf}), True))
new_orig_y_t = sp.Piecewise((y_t, t_sp < tf), 
    (y_t.subs({t_sp: tf}), True))
new_orig_th_t = sp.Piecewise((th_t, t_sp < tf), 
    (th_t.subs({t_sp: tf}), True))

# %% Test scaling robot trajectory
new_t_vals = np.linspace(to, tend_new, 300) # New time stamps
# Obstacle positions
obs_x_vals = np.array([new_obs_x_t.subs({t_sp: tv}) \
    for tv in new_t_vals], float)
obs_y_vals = np.array([new_obs_y_t.subs({t_sp: tv}) \
    for tv in new_t_vals], float)
# Original robot pose
orig_x_vals = np.array([new_orig_x_t.subs({t_sp: tv}) \
    for tv in new_t_vals], float)
orig_y_vals = np.array([new_orig_y_t.subs({t_sp: tv}) \
    for tv in new_t_vals], float)
orig_th_vals = np.array([new_orig_th_t.subs({t_sp: tv}) \
    for tv in new_t_vals], float)
# New robot x, y, theta values
x_vals = np.array([nx_t.subs({t_sp: tv}) \
    for tv in new_t_vals], float)
y_vals = np.array([ny_t.subs({t_sp: tv}) \
    for tv in new_t_vals], float)
th_vals = np.array([nth_t.subs({t_sp: tv}) \
    for tv in new_t_vals], float)
th_vals[-1] = 0.0   # Precaution (at the end of simulation)

# %%
# Show the figure
fig = plt.figure(num="Collision Avoidance", dpi=150)
ax = fig.add_subplot()
ax.set_aspect('equal')
# v_i = 100
# if True:
for v_i in range(len(new_t_vals)):  # FIXME: Don't run in VSCode
    # Reset animation
    ax.cla()
    # Show the obstacle
    obs_body = patch.Circle((obs_x_vals[v_i], obs_y_vals[v_i]), 
        obs_rad, ec='k', fc="#F06767", zorder=3.6)
    # Show the robot (original path with collision)
    rob_body_o = patch.Circle((orig_x_vals[v_i], orig_y_vals[v_i]), 
        rob_rad, ec='k', fc="#88B4E6", alpha=0.5, zorder=3.4)
    ax.plot(
        [orig_x_vals[v_i], orig_x_vals[v_i] + \
            rob_rad*np.cos(orig_th_vals[v_i])], 
        [orig_y_vals[v_i], orig_y_vals[v_i] + \
            rob_rad*np.sin(orig_th_vals[v_i])], c="#7A0C7A", 
        zorder=3.45, alpha=0.5)
    # Show the new robot path (hopefully no collision)
    rob_body = patch.Circle((x_vals[v_i], y_vals[v_i]), 
        rob_rad, ec='k', fc="#88B4E6", alpha=1, zorder=3.5)
    ax.plot(
        [x_vals[v_i], x_vals[v_i] + rob_rad*np.cos(th_vals[v_i])], 
        [y_vals[v_i], y_vals[v_i] + rob_rad*np.sin(th_vals[v_i])], 
        c="#7A0C7A", zorder=3.55)
    # Add patches
    ax.add_patch(obs_body)
    ax.add_patch(rob_body_o)
    ax.add_patch(rob_body)
    # Show the paths
    ax.plot(obs_x_vals, obs_y_vals, alpha=0.5, zorder=3)
    ax.plot(x_vals, y_vals, alpha=0.5, zorder=3)
    ax.plot(ox_i, oy_i, 'kx', zorder=3)
    # Set limits
    ax.set_xlim(start_pt[0]-5, end_pt[0]+5)
    ax.set_ylim(start_pt[1]-5, end_pt[1]+5)
    fig.savefig(f"./out/{v_i}.png")
    # Pause simulation
    # plt.pause(0.1)

# %%
