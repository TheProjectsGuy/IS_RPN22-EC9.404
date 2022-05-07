# Rule-based linear time scaling
"""
    Given a robot trajectory generated through bernstein polynomials
    (modified to return the position and velocities), and the 
    trajectory of a holonomic obstacle (straight line equation), we
    alter the robot's velocities (when the robot is in the collision
    bounds) using a user-defined linear time scaling approach.
    - The scaling is applied to robot's velocities. The scaling 's'
        is given by 's(t) = a + b*t' where 't' is the simulation time
    - For now, the script has been tested only in single collision 
        case
"""

# %% Import everything
# Main imports
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib import patches as patch
# For trajectory generation
from lib.three_point_traj_planner import NonHoloThreePtBernstein
# Utilities
import time
from tqdm import tqdm

# %%

# %%

# %% Experimental section
# Generate a random robot trajectory
# ==== Begin: User configuration area (robot trajectory) ====
# Points as [x, y]
start_pt = [0, 0]
end_pt = [50, 45]
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
t_all = sp.symbols('t') # Generic time symbol (used by functions)
print("Finding path")
# Solve for paths
x_vals, y_vals, th_vals, t_vals, x_t, y_t, th_t = \
    path_solver.solve_wpt_constr(constraint_dict)
print("Path found")
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
obs_t_col = 15      # Time of collision (for x, y intermediate)
obs_start = (4, 20)     # (x, y): Starting point of obstacle
obs_rad = 1     # Obstacle radius
rob_rad = 2.5     # Robot radius
detection_bound = 10    # Sensor for collision check (else scale = 1)
# s_func = _a + _b * t -> Functions for 's' (scaling term). t is sim.
sfunc_a = 0.001         # Constant term for s_func
sfunc_b = 0.02          # Time term for s_func
num_sim_samples = 300   # Number of time steps (not for saving!)
# ==== End: User configuration area (obstacle) ====
# Location where collision will take place
ox_i = float(x_t.subs({t_sp: obs_t_col}))
oy_i = float(y_t.subs({t_sp: obs_t_col}))
obs_x_t = obs_start[0] + ((ox_i - obs_start[0])/obs_t_col) * t_sp
obs_y_t = obs_start[1] + ((oy_i - obs_start[1])/obs_t_col) * t_sp
# Time, x, y trajectories (array) - visualize
obs_t_vals = t_vals.copy()  # np.linspace(to, tf, 100)
obs_x_vals = np.array([obs_x_t.subs({t_sp: tv}) \
    for tv in obs_t_vals])
obs_y_vals = np.array([obs_y_t.subs({t_sp: tv}) \
    for tv in obs_t_vals])

# %% Show the collision
plt.figure()
plt.title("XY plot")
plt.scatter(x_vals, y_vals, 1.0, c=t_vals)
plt.scatter(obs_x_vals, obs_y_vals, 1.0, c=t_vals)
plt.colorbar()
plt.show()

# %% Main simulation loop (with collision avoidance)
start_ctime = time.time()   # Start computer time
# Declare velocities of robot
vx_t = x_t.diff(t_sp)
vy_t = y_t.diff(t_sp)   # Get theta from velocities
# Declare velocities of obstacle
ovx_t = obs_x_t.diff(t_sp)
ovy_t = obs_y_t.diff(t_sp)
# Time values for simulation
t_sim_start, t_sim_end = to, tf
dt_sim_k1 = (t_sim_end - t_sim_start)/num_sim_samples
t_sim = t_sim_start # Current simulation time
# t_sim = 20 # Random start sim time   # FIXME: Remove this!
t_rob_local = t_sim     # Time for robot's tracking (ONLY IN SIM!)
dt_sim = dt_sim_k1  # Currently, scaling = 1
k_val = 1.0     # Value of scaling constant (for all steps)
# Pose vectors for the robot and obstacle
r_robot = [float(x_t.subs(t_sp, t_sim)), 
    float(y_t.subs(t_sp, t_sim))]
th_robot = float(th_t.subs(t_sp, t_sim))
r_obstacle = [float(obs_x_t.subs(t_sp, t_sim)), 
    float(obs_y_t.subs(t_sp, t_sim))]
# Logging variables (all time in t_sim)
robot_poses = []    # [time, x, y, theta] of the robot
obstacle_poses = [] # [time, x, y] of the obstacle
k_vals = []         # [time, k_val] - Log time scaling factor
dist_vals = []      # [time, dist_rob_obs] - Robot to obstacle
time_vals = []      # [time, t_robot_local] - Robot time (prop)
# Simulation progress bar (for robot local time)
tq_bar = tqdm(total=t_sim_end, leave=False)
# Start simulation
while t_rob_local < t_sim_end:
    # Distance between robot and obstacle ('r' vector)
    dist_ro = float(((r_robot[0] - r_obstacle[0])**2 + \
        (r_robot[1] - r_obstacle[1])**2)**0.5)
    if dist_ro < detection_bound:
        # Linear time scaling function for scaling factor
        k_val = sfunc_a + sfunc_b * t_sim
    else:
        k_val = 1.0
    # Continue robot simulation with k_val (float) scaling
    # Using velocities, progress the next states
    r_obstacle = [  # Use real time for obstacle updates
        float(r_obstacle[0] + ovx_t.subs(t_sp, t_sim) * dt_sim),
        float(r_obstacle[1] + ovy_t.subs(t_sp, t_sim) * dt_sim),
    ]
    robot_dx = float(k_val * vx_t.subs(t_sp, t_rob_local) * dt_sim)
    robot_dy = float(k_val * vy_t.subs(t_sp, t_rob_local) * dt_sim)
    r_robot = [
        float(r_robot[0] + robot_dx), float(r_robot[1] + robot_dy)
    ]
    th_robot = np.arctan2(robot_dy, robot_dx)
    # Log these values
    robot_poses.append([t_sim, r_robot[0], r_robot[1], th_robot])
    obstacle_poses.append([t_sim, r_obstacle[0], r_obstacle[1]])
    k_vals.append([t_sim, k_val])
    dist_vals.append([t_sim, dist_ro])
    time_vals.append([t_sim, t_rob_local])
    # Change in time
    t_rob_local += k_val * dt_sim   # Time scale the robot
    t_sim += dt_sim # The simulation proceeds
    tq_bar.update(k_val * dt_sim)
tq_bar.close()
# Convert all logs to numpy arrays
robot_poses = np.array(robot_poses, float)  # [time, x, y, theta]
obstacle_poses = np.array(obstacle_poses, float)    # [time, x, y]
k_vals = np.array(k_vals, float)    # [time, k_val]
dist_vals = np.array(dist_vals, float)  # [time, dist_rob_obs]
time_vals = np.array(time_vals, float)  # [time, t_robot_local]
end_ctime = time.time() # End computer time
print(f"Simulation took {end_ctime - start_ctime:.3f} seconds!")

# %% Get all trajectories (with time clipping)
# Time values
res_tvals = time_vals[:, 0]
# Robot avoiding collision
res_robposes = robot_poses[:, 1:4]  # [x, y, theta]
# Obstacle path
res_obsposes_x = np.array([obs_x_t.subs(t_sp, min(tv, tf)) \
    for tv in res_tvals], float)
res_obsposes_y = np.array([obs_y_t.subs(t_sp, min(tv, tf)) \
    for tv in res_tvals], float)
res_obsposes = np.stack([res_obsposes_x, res_obsposes_y]).T
# Robot (with collision)
res_crobotposes_x = np.array([x_t.subs(t_sp, min(tv, tf)) \
    for tv in res_tvals], float)
res_crobotposes_y = np.array([y_t.subs(t_sp, min(tv, tf)) \
    for tv in res_tvals], float)
res_crobotposes_th = np.array([th_t.subs(t_sp, min(tv, tf)) \
    for tv in res_tvals], float)
res_crobotposes = np.stack([res_crobotposes_x, res_crobotposes_y,
    res_crobotposes_th]).T
# Fix the last angle
res_robposes[-1, 2] = res_crobotposes[-1, 2]    # Theta fix
# Distance between robot and obstacle
res_cdist = np.linalg.norm(res_crobotposes[:, 0:2] - \
    res_obsposes[:, 0:2], axis=1)
res_dist = np.linalg.norm(res_robposes[:, 0:2] - \
    res_obsposes[:, 0:2], axis=1)
print(f"Processed {res_tvals.shape[0]} time samples")

# %%
# Show the time
plt.figure(figsize=(7, 3))
plt.suptitle("Time and scale")
plt.subplot(1,2,1)
plt.title("Time")
plt.xlabel("Simulation")
plt.ylabel("Robot")
plt.plot(time_vals[:, 0], time_vals[:, 1], '-')
plt.subplot(1,2,2)
plt.title("Scaling factor")
plt.xlabel("Simulation")
plt.plot(k_vals[:, 0], k_vals[:, 1], '-')
plt.tight_layout()
plt.show()
# Show the robot trajectory (avoiding collision)
plt.figure(figsize=(10, 10))
plt.suptitle("Time scaled trajectory")
plt.subplot(3,2,1)
plt.title("X")
plt.plot(res_tvals, res_robposes[:, 0], 'r-', label="Modified")
plt.plot(res_tvals, res_crobotposes[:, 0], 'r--', label="Actual")
plt.legend()
plt.subplot(3,2,3)
plt.title("Y")
plt.plot(res_tvals, res_robposes[:, 1], 'g-', label="Modified")
plt.plot(res_tvals, res_crobotposes[:, 1], 'g--', label="Actual")
plt.legend()
plt.subplot(3,2,5)
plt.title(r"$\theta$")
plt.plot(res_tvals, res_robposes[:, 2], 'b-', label="Modified")
plt.plot(res_tvals, res_crobotposes[:, 2], 'b--', label="Actual")
plt.legend()
# Obstacle trajectory
plt.subplot(3,2,2)
plt.title("Obstacle - X")
plt.plot(res_tvals, res_obsposes[:, 0], 'r-')
plt.subplot(3,2,4)
plt.title("Obstacle - Y")
plt.plot(res_tvals, res_obsposes[:, 1], 'g-')
plt.subplot(3,2,6)
plt.title("Distance")
plt.plot(res_tvals, res_dist, '-', label="Modified")
plt.plot(res_tvals, res_cdist, '--', label="Actual")
plt.axhline(obs_rad + rob_rad, ls='--', c='r')
plt.axhline(detection_bound, ls=':', c='r')
plt.legend()
# Show the plot
plt.tight_layout()
plt.show()

# %% Show as video
# Show the figure
fig = plt.figure(num="Collision Avoidance", dpi=150)
ax = fig.add_subplot()
ax.set_aspect('equal')
# v_i = 140
# if True:
for v_i in tqdm(range(len(res_tvals))): # FIXME: Don't run in VSCode
    # Reset animation
    ax.cla()
    # Show the obstacle
    obs_body = patch.Circle(
        (res_obsposes[v_i, 0], res_obsposes[v_i, 1]), 
        obs_rad, ec='k', fc="#F06767", zorder=3.6)
    # Show the robot (original path with collision)
    rob_body_o = patch.Circle(
        (res_crobotposes[v_i, 0], res_crobotposes[v_i, 1]), 
        rob_rad, ec='k', fc="#88B4E6", alpha=0.5, zorder=3.4)
    ax.plot(
        [res_crobotposes[v_i, 0], res_crobotposes[v_i, 0] + \
            rob_rad*np.cos(res_crobotposes[v_i, 2])], 
        [res_crobotposes[v_i, 1], res_crobotposes[v_i, 1] + \
            rob_rad*np.sin(res_crobotposes[v_i, 2])], c="#7A0C7A", 
        zorder=3.45, alpha=0.5)
    # Show the new robot path (hopefully no collision)
    if abs(k_vals[v_i, 1] - 1.0) > 1e-3:    # TS active
        rb_ec = 'r'
        # Line joining robot and obstacle
        ax.plot([res_robposes[v_i, 0], res_obsposes[v_i, 0]],
            [res_robposes[v_i, 1], res_obsposes[v_i, 1]], c='r',
            lw=0.2, zorder=3.65)    # Above robot and obstacle
    else:
        rb_ec = 'k'
    rob_body = patch.Circle(
        (res_robposes[v_i, 0], res_robposes[v_i, 1]), 
        rob_rad, ec=rb_ec, fc="#88B4E6", alpha=1, zorder=3.5)
    ax.plot(
        [res_robposes[v_i, 0], res_robposes[v_i, 0] + \
            rob_rad*np.cos(res_robposes[v_i, 2])], 
        [res_robposes[v_i, 1], res_robposes[v_i, 1] + \
            rob_rad*np.sin(res_robposes[v_i, 2])], 
        c="#7A0C7A", zorder=3.55)
    # Add patches
    ax.add_patch(obs_body)
    ax.add_patch(rob_body_o)
    ax.add_patch(rob_body)
    # Show the paths
    ax.plot(res_obsposes[:, 0], res_obsposes[:, 1], alpha=0.5, 
        zorder=3)
    ax.plot(res_robposes[:, 0], res_robposes[:, 1], alpha=0.5, 
        zorder=3)
    # Location where the collision will take place
    ax.plot(ox_i, oy_i, 'kx', zorder=3)
    # Set limits
    ax.set_xlim(start_pt[0]-5, end_pt[0]+5)
    ax.set_ylim(start_pt[1]-5, end_pt[1]+5)
    # Show/store result
    fig.savefig(f"./out/{v_i}.png")   # Use for saving everything
    # plt.pause(0.05)    # Use only for python script
    # plt.show()        # Use only for VSCode


# %%
