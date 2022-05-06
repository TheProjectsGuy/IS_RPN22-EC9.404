# RRT on non-holonomic differential drive robot
"""
    Running the RRT algorithm on a non-holonomic differential drive
    robot using sampling in the control space.

    Current developer: @TheProjectsGuy
"""

# %% Import everything
# Essentials
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from matplotlib import patches as pltpatch
# Utilities
import os
import abc
import time
from typing import Tuple, List
import joblib

# %% Functions
# Euclidean distance
def euclidean_dist(point_a: Tuple[float, float], 
            point_b: Tuple[float, float]) -> float:
    """
        Returns the euclidean distance between two points. Each point
        is represented as (x: float, y: float).
    """
    x1, y1 = point_a
    x2, y2 = point_b
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# %% Obstacle classes
# Obstacle base class
class Obstacle(abc.ABC):
    """
        An obstacle. This is the base class of all kinds of obstacles
        which can be placed on a map.
        Implement the following in subclasses:
        - def is_inside
            Return True if inside obstacle (on map), False if outside.
            Ideally, pass it the point as x, y.
        - def draw_on_axes(ax)
            Draws the obstacle on the axes (AxesSubplot) passed. 
            Returns nothing (axes is modified in function).
        
        The following is implemented
        - def is_outside
            Returns the negation of `is_inside`
    """
    @abc.abstractmethod
    def is_inside(self, *args, **kwargs) -> bool:
        raise NotImplementedError("Method is_inside not implemented")

    @abc.abstractmethod
    def closest_dist(self, *arg, **kwargs) -> float:
        raise NotImplementedError("Closest distance not implemented")

    @abc.abstractmethod
    def draw_on_axes(self, *args, **kwargs):
        raise NotImplementedError("Method not implemented")

    def is_outside(self, *args, **kwargs) -> bool:
        return not self.is_inside(*args, **kwargs)

# Rectangular obstacle
class Rectangle(Obstacle):
    """
        A rectangular obstacle. Initialize with start_x, start_y, 
        end_x, end_y. Note that the start point (x, y) is the lower 
        left corner and end point (x, y) is the upper right corner.
    """
    def __init__(self, start_x, start_y, end_x, end_y):
        super().__init__()
        self.sx = start_x
        self.sy = start_y
        self.ex = end_x
        self.ey = end_y

    def is_inside(self, x, y):
        """
            Checks if point (x, y) is inside the rectangle. Returns
            True if inside obstacle and False if outside.
        """
        return self.sx <= x <= self.ex and self.sy <= y <= self.ey

    def draw_on_axes(self, ax, **kwargs):
        """
            Draw obstacle on a matplotlib axes. The kwargs are
            directly passed to the pltpatch.Rectangle function
        """
        width = self.ex - self.sx
        height = self.ey - self.sy
        ax.add_patch(pltpatch.Rectangle((self.sx, self.sy), width,
            height, **kwargs))
    
    def closest_dist(self, x, y):
        """
            Returns the closest distance from the point to the sides
            of the rectangle.
        """
        min_dist = 0
        # Rectangular sides projecting out
        if self.sx <= x <= self.ex:
            min_dist = min(abs(self.sy-y), abs(self.ey-y))
        elif self.sy <= y <= self.ey:
            min_dist = min(abs(self.sx-x), abs(self.ex-x))
        else:
            min_dist = min([
                euclidean_dist((x,y), (self.sx, self.sy)),
                euclidean_dist((x,y), (self.sx, self.ey)),
                euclidean_dist((x,y), (self.ex, self.ey)),
                euclidean_dist((x,y), (self.ex, self.sy))
                ])
        if self.is_inside(x, y):
            min_dist *= -1
        return min_dist

# Circular obstacle
class Circle(Obstacle):
    """
        A circular obstacle. Initialize with center_x, center_y and
        radius.
    """
    def __init__(self, center_x, center_y, radius) -> None:
        super().__init__()
        self.cx = center_x
        self.cy = center_y
        self.r = radius

    def is_inside(self, x, y):
        """
            Checks if point (x, y) is inside the circle. Returns True
            if inside, False if outside
        """
        return euclidean_dist((x, y), (self.cx, self.cy)) <= self.r
    
    def draw_on_axes(self, ax, **kwargs):
        """
            Draw obstacle on a matplotlib axes. The kwargs are
            directly passed to the pltpatch.Rectangle function
        """
        ax.add_patch(pltpatch.Circle((self.cx, self.cy), self.r, 
            **kwargs))
    
    def closest_dist(self, x, y):
        """
            Returns the closest distance from the point to the circle
        """
        return euclidean_dist((self.cx, self.cy), (x, y)) - self.r

# %% Environment Mapping
# Main Map class for RRT
class Map:
    """
        A map of the environment with the obstacles, specially 
        designed for RRT. Currently, maps are rectangles starting at
        (0, 0) and ending at (width, height).

        Constructor parameters:
        - width: float
            Width of the environment
        - height: float
            Height of the environment
        - obstacle_list: List[Obstacle]
            A list of subclasses of Obstacle class, representing the
            obstacles in the map.
    """
    # Constructor
    def __init__(self, width: float, height: float, 
            obstacle_list: List[Obstacle]) -> None:
        self.w = width
        self.h = height
        self.obstacles = obstacle_list

    # Verify if point in free space
    def point_in_free_space(self, px: float, py: float) -> bool:
        """
            Verifies if the point passed is in the free space (out of
            all obstacles and within boundaries).

            Parameters:
            - px: float
            - py: float
                The X, Y coordinates of point

            Returns:
            - is_free: bool
                True if (X, Y) is inside the free space of the map,
                else returns False
        """
        # Check if point is inside the map
        if not (0 <= px <= self.w and 0 <= py <= self.h):
            return False
        # Check if point inside any obstacle
        for obstacle in self.obstacles:
            if obstacle.is_inside(px, py):
                return False
        # Point (px, py) is in free space
        return True
    
    # Point in occupied space
    def point_in_occupied_space(self, px: float, py: float) -> bool:
        """
            Returns the negation of `self.point_in_free_space` with
            the same arguments passed
        """
        return not self.point_in_free_space(px, py)
    
    # Draw the map
    def draw_on_axes(self, ax, line_s="-.", lim_map=False, **kwargs):
        """
            Draws the map with obstacles on the axis passed.

            Parameters:
            - ax: Axes      A matplotlib axes
            - line_s: str or None       default: "-."
                Line style of the border of the map. If None, then the
                map border is not drawn.
            - lim_map: bool         default: False
                If True, the axis limits are set to the map (fit to 
                map). If False (default behavior), the axis limits are
                set to autoscale.
            - **kwargs: All passed to obstacle `draw_on_axes` function
        """
        # Draw all obstacles
        for obstacle in self.obstacles:
            obstacle.draw_on_axes(ax, **kwargs)
        # Map boundaries
        if lim_map:
            ax.set_xlim(0, self.w)
            ax.set_ylim(0, self.h)
        else:
            # Rectangle for the map boundaries
            if line_s is not None:
                ax.add_patch(pltpatch.Rectangle((0,0), self.w, self.h,
                    fill=False, color="black", ls=line_s))
            # Autoscale
            ax.autoscale(True)

    # Find the minimum distance from obstacles
    def min_obstacles_dist(self, x, y) -> float:
        """
            Returns the minimum distance from the passed point (x, y)
            to all the obstacles in the map. Also considers the
            distance from the edges of the map. Note that the minimum
            distance is always positive.
        """
        d_v = [abs(obs.closest_dist(x, y)) for obs in self.obstacles]
        # Add the distances from the edges
        d_v.append(x)
        d_v.append(self.w - x)
        d_v.append(y)
        d_v.append(self.h - y)
        return min(d_v)

# %% Main RRT Implementation
# Maximum iteration error
class MaxIterError(Exception):
    pass

# Main RRT class
class RRTSolverDiffDrive:
    """
        A simple Rapidly exploring Random Tres solver for a two-wheel
        (LR) differential drive robot, sampling in the control space
        (for speed and angular velocity).

        Constructor parameters: 

        - map: Map
            A Map object containing the environment map (with all the
            obstacles initialized). This is the solver map which
            remains constant.
    """
    # Node class
    class NodeListItem:
        """
            A node - an item in the nodelist (a tree is mimicked by a
            list of this object). Each node has `parent` and
            `children` which stores indices. Additionally, each node
            has the following
            
            - x, y, theta: The pose of the node in the map (after
              velocity commands have been executed).
            - v, w: The velocity (linear, angular) commands to be
              executed from the `parent` to reach this node.
            
            Note: `parent` is None and `children` is an empty list by
            default. Only store index values in them.
        """
        def __init__(self, v: float, w: float, x: float, y: float, 
                theta: float) -> None:
            # Velocities to reach the node (from the parent)
            self.v = v
            self.w = w
            # Node position in the map
            self.x = x
            self.y = y
            self.theta = theta
            # Tree relations
            self.parent: int = None  # Index of parent (int)
            self.children: List[int] = []  # Indices of children

    # Constructor for the solver
    def __init__(self, map: Map) -> None:
        # Blank initialization
        self.map = map
        self.node_tree: List[RRTSolverDiffDrive.NodeListItem] = []
    
    # Clear solution
    def reset_solver(self):
        # Clear node tree
        self.node_tree: List[RRTSolverDiffDrive.NodeListItem] = []
    
    # Draw the blank map
    def draw_map_on_axes(self, ax, **kwargs):
        """
            Draws the map on the given axes `ax`. **kwargs are passed
            directly to `draw_on_axes` of the `Map` class
        """
        self.map.draw_on_axes(ax, **kwargs)
    
    # Forward kinematics of a differential drive robot
    @staticmethod
    def fk_dd(cx=0., cy=0., cth=0., vc=0., wc=0., dt=0.1):
        r"""
            Forward kinematics of a (2-wheeled) differential drive
            robot. Defaults for all paramters is 0.0, other than `dt`
            which is 0.2.

            Parameters:
            - cx, cy, cth: Current pose (x, y, theta) 
            - vc, wc: Linear and angular velocity commands
            - dt: Change in time (for next step) (default is 0.1)

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
    
    # Inverse kinematics: point & theta, point -> v, w
    @staticmethod
    def ik_dd_ptt_pt(xi, yi, thi, xj, yj, dt = 1.0):
        """
            Uses differential drive inverse kinematics to yield a
            steering command to go from one point (with known angle)
            to another point. The source point is 'i' and the
            destination point is 'j'.

            Note that the path obtained (through circle fitting) is 
            unique but the control commands can be time scaled (fast 
            travel or slow travel).

            Parameters:
            - xi, yi, thi: Initial point, with known (x, y, theta)
            - xj, yj: Destination point (x, y)
            - dt: Time scaling parameter (duration). Default = 1.0

            Returns:
            - v, w: Linear and angular velocity to reach from i -> j
            - thj: Angle at the end of the curve (after reaching 'j')
        """
        d = euclidean_dist((xi, yi), (xj, yj))
        al = math.atan2(yj-yi, xj-xi)
        w = 2*(al-thi)/dt
        v = ((al-thi)/dt)*(d/math.sin(al-thi))
        thj = 2*al - thi
        return v, w, thj
    
    # Check if the path from 'i' to 'j' is free of obstacles
    def check_path_comfortable(self, xi, yi, thi, xj, yj, thj, cv, cw,
            xg, yg, thg, dt_path = 1.0, num_steps = 5, min_dist = 5.0, 
            min_se_dist = 5.0, end_r = 3.0):
        """
            Checks if the path from a point (x_i, y_i, theta_i) to 
            point (x_j, y_j, theta_j), using the control commands 
            (cv, cw) for linear and angular velocity, can be reached.

            The `dt_path` is the time programmed to reach from 'i' to
            'j'. The `num_steps` is to adjust the discrete steps to
            check.

            Each forward kinematic solution from 'i' to 'j' (in the
            discrete time step) should be at least `min_dist` away 
            from the nearest obstacle for the path to be 'comfortable'

            The start and end points must be sufficiently apart for
            the path to be comfortable. We do not want to make quick
            turning maneuvers. This rule doesn't apply if the distance
            between the end point and the "goal" is less than the
            `min_se_dist` (else that final loop will always be 
            rejected). For this, the `end_r` and the goal is needed.

            Parameters:
            - xi, yi, thi: Starting point (x, y, theta)
            - xj, yj, thj: Ending point (x, y, theta)
            - cv, cw: Velocity commands (linear, angular)
            - xg, yg, thg: Goal point (x, y, theta)
            - dt_path: Time estimated for the entire path.
            - num_steps: The number of discrete steps to check. Make
                sure that this is fine enough to prevent very small
                obstacles in path.
            - min_dist: float   Minimum distance to be maintained from
                all the obstacles in the map (not the start to end)
            - min_se_dist: float    Minimum start to end distance for
                the path to be 'comfortable'
            - end_r: The radius for 'j' to be near goal. If the eu.
                dist. is less than `end_r`, then the `min_se_dist` is
                ignored.

            Returns:
            - pt_comfort: bool
                True if the path is comfortably away from all 
                obstacles and in the free space.
        """
        # Check minimum distance
        if euclidean_dist((xg, yg), (xj, yj)) > end_r and \
            euclidean_dist((xi, yi), (xj, yj)) < min_se_dist:
            return False    # Too close, not comfortable
        # Check traversal
        test_tvs = np.linspace(0, dt_path, num_steps)
        for t in test_tvs:
            # Run FK from 'i' to time 't' to get inertmediate point
            xm, ym, thm = self.fk_dd(xi, yi, thi, cv, cw, t)
            # Check for minimum distance
            if not (self.map.point_in_free_space(xm, ym) and \
                    self.map.min_obstacles_dist(xm, ym) > min_dist):
                # The point is not in 'comfortable' zone
                return False
        # Path in 'comfortable' zone
        return True

    # Generate a point x, y in free space
    def _generate_pt_fs(self, ex, ey):
        """
            Generates an x, y in the map which lies in the free space
        """
        # Random point
        x = random.gauss(ex, self.map.w/3.5)
        y = random.gauss(ey, self.map.h/3.5)
        # x = random.uniform(0, self.map.w)
        # y = random.uniform(0, self.map.h)
        # Check if in free space
        if self.map.point_in_free_space(x, y):
            return (x, y)
        else:
            # Rerun the function again
            # return (self.map.w/2, self.map.h/2) # Testing!
            return self._generate_pt_fs(ex, ey)

    # Get addable point in the map
    def _get_addable_point(self, rand_pt, curr_pt, cut_dist):
        """
            Get a point from the current point in the direction of the
            random point, with a maximum distance. This function does
            not check if the returned point is in the free space of
            the map.

            Parameters:
            - rand_pt: (float, float)
                Random point that was generated
            - curr_pt: (float, float)
                Current point
            - cut_dist: float
                The maximum distance for adding the node
            
            Returns:
            - x, y: (float, float): The point that can be added
        """
        r = euclidean_dist(rand_pt, curr_pt)
        if r < cut_dist:
            # Within the cut distance
            return rand_pt
        else:
            # Outside the cut distance
            theta = math.atan2(rand_pt[1]-curr_pt[1], 
                rand_pt[0]-curr_pt[0])  # Angle
            add_pt = (  # Point to add (x, y)
                cut_dist*math.cos(theta)+curr_pt[0],
                cut_dist*math.sin(theta)+curr_pt[1])
            return add_pt

    # Find closest index in the node tree
    def _closest_index(self, pt_x, pt_y):
        # Numpy array
        np_arr = np.array([[n.x, n.y] for n in self.node_tree])
        # Closest index to node from the point
        pt = np.array([pt_x, pt_y])
        # Closest point (index) in the node_tree
        return int(np.argmin(np.linalg.norm(np_arr-pt, axis=1)))

    # Get addable point in the map
    def _get_addable_point(self, rand_pt, curr_pt, cut_dist):
        """
            Get a point from the current point in the direction of the
            random point, with a maximum distance. This function does
            not check if the returned point is in the free space of
            the map.

            Parameters:
            - rand_pt: (float, float)
                Random (x, y) point that was generated
            - curr_pt: (float, float)
                Current point (x, y)
            - cut_dist: float
                The maximum distance for adding the node
            
            Returns:
            - x, y: (float, float): The point that can be added
        """
        r = euclidean_dist(rand_pt, curr_pt)
        if r < cut_dist:
            # Within the cut distance
            return rand_pt
        else:
            # Outside the cut distance
            theta = math.atan2(rand_pt[1]-curr_pt[1], 
                rand_pt[0]-curr_pt[0])  # Angle
            # Point to add (x, y)
            add_x = cut_dist*math.cos(theta)+curr_pt[0]
            add_y = cut_dist*math.sin(theta)+curr_pt[1]
            return add_x, add_y

    # Single stage draw output
    def _draw_single_stage(self, st_pt, en_pt, rnd_pt, out_file, en):
        """
            Draw the entire map - with start, end, and tree, and save
            it to a file. The folder path for file should be existing.

            Parameters:
            - st_pt: (float, float)     The (x, y) starting point
            - en_pt: (float, float)     The (x, y) ending point
            - rnd_pt: (float, float)     The (x, y) random point
            - out_file: str
                The output file where the image is stored
            - en: int       Number of intermediary points for edges
        """
        # Axes
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = fig.add_subplot()
        ax.set_aspect('equal', 'box')
        # Obstacles in the map (in red)
        self.map.draw_on_axes(ax, "-.", color='r')
        # Start and end star markers
        ax.plot(st_pt[0], st_pt[1], "b*", ms=10.0)
        ax.plot(en_pt[0], en_pt[1], "g*", ms=10.0)
        # Draw tree
        for node in self.node_tree:
            # Nodes (in grey)
            ax.plot(node.x, node.y, 'o', c='grey', ms=2.0)
            # Child connections
            for c in node.children: # 'c' is index
                xp, yp = node.x, node.y
                # Get five intermediates and join
                for t in np.linspace(0, 1.0, en):
                    xi, yi, _ = self.fk_dd(node.x, node.y, node.theta,
                        self.node_tree[c].v, self.node_tree[c].w, t)
                    # Draw it with lines
                    plt.plot([xi, xp], [yi, yp], '--', c='grey')
                    # Remember this point
                    xp, yp = xi, yi
        # Random point generated
        ax.plot(rnd_pt[0], rnd_pt[1], "mo", ms=5.0)
        # Save as file name
        fig.savefig(out_file)
        # Clear the figure and buffer
        fig.clear()
        plt.close('all')

    # Solve the entire RRT problem
    def solve(self, start_pt, end_pt, end_r = 4.0, max_dist = 10.0,
            dt_edge = 1.0, edge_nst = 4, min_obs_dist = 4.0, 
            max_iter = 1e6, min_edge_dist = 2.0, out_dir = "./out"):
        """
            Solve the RRT problem. Pass the start and end points and
            some solution parameters and the function solves for an
            RRT path (using sampling in control space).

            Parameters:
            - start_pt: The starting point on map (x, y, theta)
            - end_pt: The ending point on map (x, y, theta)
            - end_r: float      default: 3.0
                The algorithm terminates when a node is found within 
                this distance from the goal.
            - max_dist: float       default: 10.0
                The maximum distance between nodes to be added in the
                tree.
            - dt_edge: float        default: 1.0
                The time step for traversing edges (same for all 
                edges).
            - edge_nst: int         default: 4
                The number of steps to take in edges for checking if
                the path is "comfortable"
            - min_obs_dist: float       default: 4.0
                The minimum distance paths should maintain from the
                obstacles.
            - max_iter: int         default: 1e6
                The maximum number of iterations to try. If the loop
                runs more number of iterations than this, then a 
                `MaxIterError` is thrown.
            - min_edge_dist: float      default: 2.0
                The minimum distance between nodes being added
            - out_dir: str          default: "./out"
                The output directory for the images. If None, then
                nothing is saved. The images are numbered in a 
                sequence.

            Returns:
            - xyth_vals: np.ndarray     shape: (N+1), 3
                The x, y, theta values for the path (including the 
                starting point)
            - vw_vals: np.ndarray       shape: N, 2
                The linear and angular velocity controls (v, w) for
                the robot to apply. The 'i'th index is for the 
                controls to apply to the 'i'th index in `xyth_vals`.
        """
        # Start fresh
        self.reset_solver()
        iter_num = 0        # Failsafe. Do NOT have more iterations
        out_fi = 0          # Output file index
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
        # Local variables
        sx, sy, sth = start_pt  # Start point
        ex, ey, eth = end_pt    # End point
        NLI = RRTSolverDiffDrive.NodeListItem   # Shorthand
        # Add starting node to tree
        node = NLI(v=0, w=0, x=sx, y=sy, theta=sth)
        self.node_tree.append(node)
        # Till the last node added is not close enough
        while euclidean_dist((node.x, node.y), (ex, ey)) > end_r:
            # Generate a random point in the free space
            randx, randy = self._generate_pt_fs(ex, ey)
            # Node in tree closest to random (x, y); node 'i'
            ci = self._closest_index(randx, randy)
            nd_i = self.node_tree[ci]   # Node at closest in tree
            # Get intermediate point that can be added to tree
            ax, ay = self._get_addable_point((randx, randy), 
                (nd_i.x, nd_i.y), max_dist) # Addable point - 'j'
            # Get command to reach from 'i' to 'j', and theta at 'j'
            cv, cw, ath = self.ik_dd_ptt_pt(nd_i.x, nd_i.y, 
                nd_i.theta, ax, ay, dt_edge)
            # Check if path from 'i' to 'j' is "comfortable"
            if self.check_path_comfortable(nd_i.x, nd_i.y, nd_i.theta,
                    ax, ay, ath, cv, cw, ex, ey, eth, dt_edge, 
                    edge_nst, min_obs_dist, min_edge_dist, end_r):
                # Add the 'j' node to this tree (list)
                node = NLI(cv, cw, ax, ay, ath)
                node.parent = ci
                # This point will have the last index (current len)
                self.node_tree[ci].children.append(
                    len(self.node_tree))    # This 'j' is a child
                # Add to tree
                self.node_tree.append(node)
                # Log the modification to tree
                if out_dir is not None:
                    # Save output file
                    self._draw_single_stage(start_pt, end_pt, (randx, 
                        randy), f"{out_dir}/{out_fi}.png", edge_nst)
                    out_fi += 1
            # Next iteration
            iter_num += 1
            if iter_num >= max_iter:
                raise MaxIterError(f"Iter: {max_iter}")
        # Backtrack from the last node to the start
        xyth_vals = []  # Store (x, y, theta) till start
        vw_vals = []    # Store command v, w values
        cnode = self.node_tree[-1]  # Start with last node (goal)
        while cnode.parent is not None: # Till start
            xyth_vals.append([cnode.x, cnode.y, cnode.theta])
            vw_vals.append([cnode.v, cnode.w])  # To parent
            cnode = self.node_tree[cnode.parent]    # Parent
        # Add the starting point
        xyth_vals.append([sx, sy, sth])
        # Convert to numpy arrays
        xyth_vals: np.ndarray = np.array(xyth_vals)
        vw_vals: np.ndarray = np.array(vw_vals)
        # Return trajectory (points and commands)
        return xyth_vals[::-1], vw_vals[::-1]


# %%

# %% Experimental section

# %% Main function
if __name__ == "__main__":
    # %% Initialize obstacles
    rect1 = Rectangle(40, 62, 60, 100)
    rect2 = Rectangle(70, 40, 80, 60)
    rect3 = Rectangle(40, 0, 60, 40)
    cir1 = Circle(20, 20, 7)
    cir2 = Circle(77, 82, 3)
    cir3 = Circle(90, 70, 2)
    rrt_map = Map(100, 100, [rect1, rect2, rect3, cir1, cir2, cir3])
    rrt_solver = RRTSolverDiffDrive(rrt_map)
    start_pose = (5, 5, 0)
    end_pose = (95, 95, 0)
    print(f"Starting the RRT solver")

    start_time = time.time()
    try:
        rrt_path = rrt_solver.solve(start_pose, end_pose, max_dist=35, 
            edge_nst=20, min_edge_dist=5, min_obs_dist=2.0, 
            end_r=2.0)
    except KeyboardInterrupt:
        print(f"Tree has {len(rrt_solver.node_tree)} nodes!")
        exit(1)
    end_time = time.time()
    print(f"It took {end_time-start_time:.3f} seconds!")
    xyth_vals, vw_vals = rrt_path[0], rrt_path[1]   # Extract points

    # %% Visualize the result
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
            for t in np.linspace(0, 1.0, 20):  # Fine resolution here
                xi, yi, _ = rrt_solver.fk_dd(node.x, node.y, 
                    node.theta, rrt_solver.node_tree[c].v, 
                    rrt_solver.node_tree[c].w, t)
                # Draw it with lines
                plt.plot([xi, xp], [yi, yp], '--', c='grey')
                # Remember this point
                xp, yp = xi, yi
    fig.set_tight_layout(True)
    try:
        plt.show(fig)
    except:
        plt.show()

    # Save the paths and the tree
    out_file = "./output_cache"
    result = {
        "solver": rrt_solver,
        "start_pose": start_pose,
        "end_pose": end_pose,
        "result": rrt_path,
        "map": rrt_map,
    }
    joblib.dump(result, out_file, 1)
    print(f"Cache result saved to file {out_file}")

# %%


