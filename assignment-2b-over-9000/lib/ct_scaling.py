# Constant time scaling functions
"""
"""

# %% Import everything
import sympy as sp
import numpy as np

# %%
def sp_time_scaling_eq(eq_t: sp.Expr, t_start: float, t_end: float, 
        t_lim: list, k = 1/2):
    """
        Apply constant time scaling to derivative (velocity). The
        function applies time scaling to the `eq_t` function, and
        returns the new `eq_t`. For sympy expression based time 
        functions only.

        Parameters:
        - eq_t: sp.Expr
            Equation of time. Should have `t` as the only free symbol
        - t_start, t_end: float
            The start and end of time scaling. The time between this
            zone is stretched (along with reduced velocity).
        - t_lim: list       The time limits of the simulation. Used
            to return the new end time (for simulation)
        - k: float          The scaling factor for reduction of 
            velocity and increase of time gap (between `t_start` and
            `t_end`)
        
        Returns:
        - neq_t: sp.Expr
            New `eq_t` (time scaled) with `t` as only free symbol
        - t2_new: float
            The new value of `t_end` (end of time scaling session)
        - tend_new: float
            The new end time for the simulation
    """
    # Time variable
    t = sp.symbols('t', positive=True, real=True)
    assert len(eq_t.free_symbols) == 1, eq_t.free_symbols.pop() == t
    # Time derivative (which will be scaled)
    deq_t = eq_t.diff(t)    # Bias @ t = 0 lost!
    # Time values
    t_c1 = t_start
    t_c2 = t_end
    t1_new = t_c1
    t2_new = ((t_c2-t_c1)/k) + t_c1
    tend_new = t2_new + (t_lim[1] - t_c2)
    del_t2 = t2_new - t_c2  # Shift for the end part
    # New equation (for time scaled derivative)
    new_deq_t = sp.Piecewise((deq_t, t < t1_new), 
        (k*deq_t.subs({t: k*(t-t_c1)+t_c1}), t < t2_new),
        (deq_t.subs({t: t-del_t2}), True))
    # New equation for normal variable (add bias which we loose in dt)
    neq_t = sp.integrate(new_deq_t) + eq_t.subs({t:0})
    return neq_t, t2_new, tend_new

# %%
