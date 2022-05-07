# Testing time scaling
"""
    Testing how velocity time scaling can be accomplished
"""

# %%
import sympy as sp
import numpy as np
from matplotlib import pyplot as plt
from lib.ct_scaling import sp_time_scaling_eq

# %%
# Define trajectory
t = sp.symbols('t', positive=True, real=True)
x_t = t + 4
y_t = 2 * t**2 + 2 * x_t + 4
t_lim = [0, 10]

# Show the x, y plot
t_vals = np.linspace(t_lim[0], t_lim[1], 100)
x_vals = np.array([x_t.subs({t: tv}) for tv in t_vals], float)
y_vals = np.array([y_t.subs({t: tv}) for tv in t_vals], float)
# Show the figure
plt.figure(figsize=(6.4, 7.5))
plt.subplot(2,1,1)
plt.title("X")
plt.plot(t_vals, x_vals)
plt.subplot(2,1,2)
plt.title("Y")
plt.plot(t_vals, y_vals)
plt.show()
plt.figure(figsize=(7, 7))
plt.scatter(x_vals, y_vals, c=t_vals, s=1.0)
plt.colorbar()
plt.show()

# %%
k = 1/2
t_c1 = 4
t_c2 = 5
plt.figure(figsize=(6.4, 7.5))
plt.subplot(2,1,1)
nx1_t, t2_new, tend_new = sp_time_scaling_eq(x_t, t_c1, t_c2, t_lim,
    k)
nt1_vals = np.linspace(t_lim[0], tend_new, 100)
nx1_vals = np.array([nx1_t.subs({t: tv}) for tv in nt1_vals], float)
plt.title("X")
plt.plot(nt1_vals, nx1_vals, label="new")
plt.plot(t_vals, x_vals, '--', label="old")
plt.axvline(t_c1, color='k', ls='--')
plt.axvline(t_c2, color='k', ls=':')
plt.axvline(t2_new, color='k', ls=':')
plt.axhline(x_t.subs(t, t_c2), color='r', ls='-')
plt.axhline(x_t.subs(t, t_lim[1]), color='y', ls='-')
plt.legend()
plt.subplot(2,1,2)
ny1_t, t2_new, tend_new = sp_time_scaling_eq(y_t, t_c1, t_c2, t_lim,
    k)
nt1_vals = np.linspace(t_lim[0], tend_new, 100)
ny1_vals = np.array([ny1_t.subs({t: tv}) for tv in nt1_vals], float)
plt.title("Y")
plt.plot(nt1_vals, ny1_vals, label="new")
plt.plot(t_vals, y_vals, '--', label="old")
plt.axvline(t_c1, color='k', ls='--')
plt.axvline(t_c2, color='k', ls=':')
plt.axvline(t2_new, color='k', ls=':')
plt.axhline(y_t.subs(t, t_c2), color='r', ls='-')
plt.axhline(y_t.subs(t, t_lim[1]), color='y', ls='-')
plt.legend()
plt.show()

# %% Check implementation with actual
dx_t = x_t.diff(t)
dy_t = y_t.diff(t)
t1_new = t_c1
t2_new = ((t_c2-t_c1)/k) + t_c1
tend_new = t2_new + (t_lim[1] - t_c2)
del_t2 = t2_new - t_c2  # Shift for the end part
# New equations (after time scaling)
new_dx_t = sp.Piecewise((dx_t, t < t1_new), 
    (k*dx_t.subs({t: k*(t-t_c1)+t_c1}), t < t2_new),
    (dx_t.subs({t: t-del_t2}), True))
new_dy_t = sp.Piecewise((dy_t, t < t1_new), 
    (k*dy_t.subs({t: k*(t-t_c1)+t_c1}), t < t2_new),
    (dy_t.subs({t: t-del_t2}), True))

nx_t = sp.integrate(new_dx_t) + x_t.subs({t:0})
ny_t = sp.integrate(new_dy_t) + y_t.subs({t:0})
nt_vals = np.linspace(t_lim[0], tend_new, 100)
nx_vals = np.array([nx_t.subs({t: tv}) for tv in nt_vals], float)
ny_vals = np.array([ny_t.subs({t: tv}) for tv in nt_vals], float)
plt.figure(figsize=(6.4, 7.5))
plt.subplot(2,1,1)
plt.title("X")
plt.plot(nt_vals, nx_vals, label="new")
plt.plot(t_vals, x_vals, '--', label="old")
plt.axvline(t_c1, color='k', ls='--')
plt.axvline(t_c2, color='k', ls=':')
plt.axvline(t2_new, color='k', ls=':')
plt.axhline(x_t.subs(t, t_c2), color='r', ls='-')
plt.axhline(x_t.subs(t, t_lim[1]), color='y', ls='-')
plt.legend()
plt.subplot(2,1,2)
plt.title("Y")
plt.plot(nt_vals, ny_vals, label="new")
plt.plot(t_vals, y_vals, '--', label="old")
plt.axvline(t_c1, color='k', ls='--')
plt.axvline(t_c2, color='k', ls=':')
plt.axvline(t2_new, color='k', ls=':')
plt.axhline(y_t.subs(t, t_c2), color='r', ls='-')
plt.axhline(y_t.subs(t, t_lim[1]), color='y', ls='-')
plt.legend()
plt.show()

# %%
