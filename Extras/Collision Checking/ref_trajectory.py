import numpy as np


v = 1
dt = 0.1
num_st_pts = int(v/dt)
num_pts = 50

def cubic_spiral(theta_i, theta_f, n=10):
  x = np.linspace(0, 1, num=n)
  #-2*x**3 + 3*x**2
  return (theta_f-theta_i)*(-2*x**3 + 3*x**2) + theta_i
    
def straight(dist, curr_pose, n=num_st_pts):
  # the straight-line may be along x or y axis
  x0, y0, t0 = curr_pose
  xf, yf = x0 + dist*np.cos(t0), y0 + dist*np.sin(t0)
  x = (xf - x0) * np.linspace(0, 1, n) + x0
  y = (yf - y0) * np.linspace(0, 1, n) + y0
  return x, y, t0*np.ones_like(x)

def turn(change, curr_pose, n=num_pts):
  # adjust scaling constant for desired turn radius
  x0, y0, t0 = curr_pose
  theta = cubic_spiral(t0, t0 + np.deg2rad(change), n)
  x= x0 + np.cumsum(v*np.cos(theta)*dt)
  y= y0 + np.cumsum(v*np.sin(theta)*dt)
  return x, y, theta

def generate_trajectory(route, init_pose = (0, 0,np.pi/2)):
  curr_pose = init_pose
  func = {'straight': straight, 'turn': turn}
  x, y, t = np.array([]), np.array([]),np.array([])
  for manoeuvre, command in route:
    px, py, pt = func[manoeuvre](command, curr_pose)
    curr_pose = px[-1],py[-1],pt[-1]  # New current pose
    x = np.concatenate([x, px])
    y = np.concatenate([y, py])
    t = np.concatenate([t, pt])
        
  return x, y, t
