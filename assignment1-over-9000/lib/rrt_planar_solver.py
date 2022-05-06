# RRT implementation for the planar environment
"""
    Implementation of a basic RRT (Rapidly exploring Random Trees)
    algorithm.

    The algorithm has the following procedures
    1. Create a map with `Obstacle` objects. Currently only 
        `Rectangle` and `Circle` are supported.
    2. Initialize the `SolverRRT` with the map (which will remain 
        static throughout). Multiple `solve` calls with different
        start and end points can be made. The following distances are
        to be kept in mind
        - end_range: Stop when goal in this range
        - obj_dist: Minimum distance from obstacles
        - max_dist: Maximum distance between new nodes to be added
    3. The function `save_results_figs` can be called to save the path
        as it is to be traversed in the tree.
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
            to all the obstacles in the map. Note that the minimum
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
class SolverRRT:
    """
        A simple Rapidly exploring Random Trees solver, straight 
        forward sampling. The map remains constant.

        Constructor parameters:
        - map: Map
            A Map object containing the environment map (with all the
            obstacles initialized). This is the solver map which
            remains constant.
    """
    # Node class
    class NodeListItem:
        """
            A node - an item in the nodelist. Try to mimmick a tree
            through a list of this object. Each node has a `parent`
            and `children` which store the index (in the list) of the
            corresponding parent and children nodes in the tree. The
            list should therefore contain all the nodes in the tree.
            Note that the `parent` is None by default, and `children`
            is an empty list by default. Only indices must be stored
            in them.

            Constructor parameters:
            - x, y: float       The x, y point values of the node
        """
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y
            self.parent = None  # Index of parent (int)
            self.children = []  # Index of the children (ints)

    # Constructor for the solver
    def __init__(self, map: Map) -> None:
        # Blank initialization
        self.map = map
        self.node_tree: List[SolverRRT.NodeListItem] = []
    
    # Clear solution
    def reset_solver(self):
        # Clear the node tree, so that a new query can be made
        self.node_tree: List[SolverRRT.NodeListItem] = []
    
    # Draw the blank map
    def draw_map_on_axes(self, ax, **kwargs):
        """
            Draws the map on the given axes `ax`. **kwargs are passed
            directly to `draw_on_axes` of the `Map` class
        """
        self.map.draw_on_axes(ax, **kwargs)
    
    # Find closest index in the node tree
    def _closest_index(self, pt_x, pt_y):
        # Numpy array
        np_arr = np.array([[n.x, n.y] for n in self.node_tree])
        # Closest index to node from the point
        pt = np.array([pt_x, pt_y])
        # Closest point (index) in the node_tree
        return int(np.argmin(np.linalg.norm(np_arr-pt, axis=1)))

    # Generate a random node in free space
    def _gen_free_pt(self):
        # Random point
        x = random.uniform(0, self.map.w)
        y = random.uniform(0, self.map.h)
        if self.map.point_in_free_space(x, y):
            return (x, y)
        else:
            return self._gen_free_pt()

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

    # Single stage draw output
    def _draw_single_stage(self, st_pt, en_pt, rnd_pt, out_file):
        """
            Draw the entire map - with start, end, and tree, and save
            it to a file. The file should be existing

            Parameters:
            - st_pt: (float, float)     The (x, y) starting point
            - en_pt: (float, float)     The (x, y) ending point
            - rnd_pt: (float, float)     The (x, y) random point
            - out_file: str
                The output file where the image is stored
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
        # Tree
        for node in self.node_tree:
            # Nodes (in grey)
            ax.plot(node.x, node.y, 'o', c='grey', ms=2.0)
            # Child connections
            for i in node.children:
                ax.plot([node.x, self.node_tree[i].x], 
                    [node.y, self.node_tree[i].y], '--', c='grey')
        # Random point generated
        ax.plot(rnd_pt[0], rnd_pt[1], "mo", ms=5.0)
        # Save as file name
        fig.savefig(out_file)
        # Clear the figure and buffer
        fig.clear()
        plt.close('all')

    # Do the entire RRT problem
    def solve(self, start_pt, end_pt, end_range=5.0, max_dist=2.0, 
            obj_dist=5.0, max_iter=1e5, out_dir = "./out"):
        """
            Solve the RRT problem. Give the starting point and the
            ending point (goal). The function resets the existing tree
            through `reset_solver`. The `out_dir` stores images of
            actions (modifications to the tree).

            Parameters:
            - start_pt: (float, float)
                The starting point on the map
            - end_pt: (float, float)
                The ending point on the map
            - end_range: float      default: 4.0
                The range to terminate the search. The last node is
                the end_pt. The second last node will be within this
                distance from the end_pt. Search terminates when a
                node is found within this distance.
            - obj_dist: float       default: 5.0
                The minimum distance the nodes added to the path
                should maintain from the obstacles in the map.
            - max_dist: float       default: 2.0
                The maximum distance between nodes to be added in the
                tree.
            - max_iter: int         default: 1e5
                The maximum number of iterations to try. If the loop
                runs more number of iterations than this, then a 
                `MaxIterError` is thrown.
            - out_dir: str          default: "./out"
                The output directory for the images. If None, then
                nothing is saved. The images are numbered in a 
                sequence.
            
            Returns:
            - point_vals: np.ndarray        shape: n1, 2
                A path of nodes (x, y) that can be followed to go from
                start to end. These nodes are a part of the 
                self.node_tree variable, and can be considered as 
                waypoints.            

            Throws:
            - FileExistsError: `out_dir` exists
            - MaxIterError: `max_iter` limit crossed
        """
        self.reset_solver() # Start fresh
        # Local variables
        iter_num = 0    # Do NOT go beyond max_iter
        out_findex = 0  # Output file index
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
        # Add starting node to tree
        node = SolverRRT.NodeListItem(start_pt[0], start_pt[1])
        self.node_tree.append(node) # Always add `node`
        # Till last added node is not close enough
        while euclidean_dist((node.x, node.y), end_pt) > end_range:
            # Generate a random node in free space
            x, y = self._gen_free_pt()
            # Find node in tree that is closest to this point
            ci = self._closest_index(x, y)
            nd_i = self.node_tree[ci]   # Node at 'i'
            # Get the point that can be added to the tree
            ax, ay = self._get_addable_point((x, y), (nd_i.x, nd_i.y),
                max_dist)
            # If point in free space, add it
            if self.map.point_in_free_space(ax, ay) \
                and self.map.min_obstacles_dist(ax, ay) > obj_dist:
                # print(f"[{ci}] Adding ({ax}, {ay}), "
                #     f"{len(self.node_tree)}")
                # Create a new node
                node = SolverRRT.NodeListItem(ax, ay)
                node.parent = ci    # Came from this 'ci' node
                # Add the point to node[ci] in the tree (child)
                self.node_tree[ci].children.append(
                    len(self.node_tree))    # Will be the last item
                # Add to tree (last item)
                self.node_tree.append(node)
                # Output logging
                if out_dir is not None:
                    # Save output file
                    self._draw_single_stage(start_pt, end_pt, (x, y),
                        f"{out_dir}/{out_findex}.png")
                    out_findex += 1
            # Next iteration
            iter_num += 1
            if iter_num >= max_iter:
                raise MaxIterError(f"Iter: {max_iter}")
        # `node` was last added, closest to goal
        # Add the goal as a node (parent is the present `node`)
        p_i = len(self.node_tree)    # Current node
        end_node = SolverRRT.NodeListItem(end_pt[0], end_pt[1])
        end_node.parent = p_i-1 # Last node in list (currently)
        self.node_tree[p_i-1].children.append(p_i)  # End node
        self.node_tree.append(end_node)
        # Back track from end to start using parent
        n_i_vals = [p_i]   # Store indices
        cnode = self.node_tree[-1]  # End node (goal)
        while cnode.parent is not None: # Till we reach start node
            n_i_vals.append(cnode.parent)
            cnode = self.node_tree[cnode.parent]
        n_i_vals:list[int] = list(reversed(n_i_vals)) # Start to end
        point_vals = np.array([     # Starting to end [x, y] points
            [self.node_tree[i].x, self.node_tree[i].y] \
                for i in n_i_vals])
        return point_vals
    
    # Save results
    def save_results_figs(self, points, st_pt, en_pt, out_dir):
        """
            Save the path as it goes in points.
        """
        out_dir = os.path.realpath(os.path.expanduser(out_dir))
        # Check output directory
        if not os.path.isdir(out_dir):
            # Make directory
            print(f"[INFO]: Output directory '{out_dir}' created")
            os.makedirs(out_dir)
        else:
            print(f"[ERROR]: Output directory exists")
            raise FileExistsError(f"Output directory: {out_dir}")
        # Multiple figures (number of points)
        for i in range(len(points)+1):
            # Main figure
            fig = plt.figure(figsize=(8, 8), dpi=300)
            ax = fig.add_subplot()
            ax.set_aspect('equal', 'box')
            # Main map
            self.map.draw_on_axes(ax, "-.", color='r')
            # Start and end star markers
            ax.plot(st_pt[0], st_pt[1], "b*", ms=10.0)
            ax.plot(en_pt[0], en_pt[1], "g*", ms=10.0)
            # Tree of all nodes
            for node in self.node_tree:
                # Nodes (in grey)
                ax.plot(node.x, node.y, 'o', c='grey', ms=2.0)
                # Child connections
                for j in node.children:
                    ax.plot([node.x, self.node_tree[j].x], 
                        [node.y, self.node_tree[j].y], '--', c='grey')
            # Points 0 to i
            ax.plot(points[0:i,0], points[0:i,1], 'co-')
            # Save as file name
            # print(f"Saving: {i}.png")
            fig.savefig(f"{out_dir}/{i}.png")
            # Clear the figure and buffer
            fig.clear()
            plt.close('all')

# %% Main function
if __name__ == "__main__":
    # %% Initialize obstacles
    rect1 = Rectangle(40, 62, 60, 100)
    rect2 = Rectangle(70, 40, 80, 60)
    rect3 = Rectangle(40, 0, 60, 40)
    cir1 = Circle(20, 20, 7)
    cir2 = Circle(77, 82, 3)
    cir3 = Circle(90, 70, 2)

    # %% Main RRT implementation
    rrt_map = Map(100, 100, [rect1, rect2, rect3, cir1, cir2, cir3])
    rrt_solver = SolverRRT(rrt_map)
    start_point = (5, 5)
    end_point = (95, 95)
    print("Starting the RRT solver")
    # %% Solve RRT
    start_time = time.time()
    rrt_path = rrt_solver.solve(start_point, end_point, max_dist=5)
    end_time = time.time()
    print(f"It took {end_time-start_time:.6f} seconds!")
    print(f"Explored {len(rrt_solver.node_tree)} nodes and found a "
        f"path with {rrt_path.shape[0]} points")
    # %% Plotting
    rrt_solver.save_results_figs(rrt_path, start_point, end_point, 
        "./path_out")
    print(f"All files saved")

# %%

# %% Experiments section
