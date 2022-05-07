# Plan trajectory using three point constraints
r"""
    Uses Bernstein polynomials to plan trajectory of a non-holonomic
    agent.

    Uses the user defined library for calculating Bernstein 
    polynomials.
"""

# %% Import everything
import numpy as np
import sympy as sp
# Functions to get Bernstein basis and derivatives
try:
    from bernstein_basis import get_ts_bernstein_basis_diff
except ModuleNotFoundError:
    from lib.bernstein_basis import get_ts_bernstein_basis_diff
# For comparison
try:
    from starter_code import get_product_functions
except ModuleNotFoundError:
    from lib.starter_code import get_product_functions
# Plotting in this file
from matplotlib import pyplot as plt


# %% Classes
class NonHoloThreePtBernstein:
    r"""
        Class to handle non-holonomic trajectory using three points
        and using Bernstein approximation. Needs the following
        - x and y values for t = to, tc, tf
        - \dot{x} and \dot{y} values for t = to, tc, tf
        - \theta values for t = to and tf

        ==== Constructor ====

        Creates a NonHoloThreePtBernstein object.

        Parameters:
        - n_val: int    default: 5
            The degree of the Bernstein polynomial to use for modeling
            the trajectories.
    """
    # Constructor
    def __init__(self, n_val = 5) -> None:
        self.n = n_val
        # Bernstein basis and derivative coefficients
        self.bb_tval, self.dbb_tval = \
            get_ts_bernstein_basis_diff(n_val)
        # Time, end, start, wpt.
        self.t, self.t_f, self.t_o, self.t_w = \
            sp.symbols("t, t_f, t_o, t_w")
        # Weights for bernstein basis
        self.wx = [sp.symbols(f"w_{{x_{i}}}") \
            for i in range(n_val+1)]    # X
        self.wk = [sp.symbols(f"w_{{k_{i}}}") \
            for i in range(n_val+1)]    # K
        # Models
        self.mx: sp.Expr = 0     # Model for x
        self.mdx: sp.Expr = 0    # Model for dx
        self.mk: sp.Expr = 0     # Model for k=tan(theta)
        self.mdk: sp.Expr = 0    # Model for dk=w(1+sec^2 (theta))
        for i in range(n_val+1):
            self.mx += self.wx[i] * self.bb_tval[i]    # For 'x'
            self.mdx += self.wx[i] * self.dbb_tval[i]  # For 'dx'
            self.mk += self.wk[i] * self.bb_tval[i]    # For 'k'
            self.mdk += self.wk[i] * self.dbb_tval[i]  # For 'dk'
        # ==== Substitution variables ====
        # X value of path - Start, waypoint, final
        self.xo, self.xw, self.xf = sp.symbols(r"x_o, x_w, x_f")
        self.dxo, self.dxw, self.dxf = \
            sp.symbols(r"\dot{x}_o, \dot{x}_w, \dot{x}_f")  # dx
        # Y values - Start, waypoint, final
        self.yo, self.yw, self.yf = sp.symbols(r"y_o, y_w, y_f")
        # Theta (heading) of path (in rad.)
        self.tho, self.thw, self.thf = \
            sp.symbols(r"\theta_o, \theta_w, \theta_f")
        # k = tan(theta) for each theta
        self.ko, self.kw, self.kf = map(sp.tan, 
            (self.tho, self.thw, self.thf))

    # Solve through waypoint constraint
    def solve_wpt_constr(self, constrs :dict = dict(), 
            t_nsteps = 100) -> tuple:
        r"""
            Given the constraints (including waypoints), solve for
            the path and return the x, y, theta values for the curve.
            The path consists of only one waypoint and there is no
            obstacle checking.
            Currently, only the following constraints are used
            - For 'wx': x(to), x(tw), x(tf), dx(to), dx(tw), dx(tf)
            - For 'wk': y(to), y(tw), y(tf), k(to), k(tf), dk(to), 
                        dk(tf)
            
            Where k = tan(theta), and $\theta$ is the heading. The
            constraints should be given in a dictionary.

             Returns the x, t, theta values (time sampled), as well as
            sympy functions (where time can be substituted). The time
            values `t_vals` is also sent for synchronizing trajectory.

            Parameters:
            - constrs: dict     default: dict()
                A dictionary (str->float) containing constraints
                - "to": Starting time       default: 0
                - "tw": Waypoint time       default: 20
                - "tf": Ending time         default: 50
                - "xo": Starting 'x' value      default: 0
                - "xw": Waypoint 'x' value      default: 5
                - "xf": Ending 'x' value        default: 40
                - "yo": Starting 'y' value      default: 0
                - "yw": Waypoint 'y' value      default: 30
                - "yf": Ending 'y' value        default: 40
                - "ko": Starting 'k' value      default: 0
                - "kw": Waypoint 'k' value      default: np.pi/5
                - "kf": Ending 'k' value        default: 0
                - "dxo": Starting 'dx' value        default: 0
                - "dxw": Waypoint 'dx' value        default: 0
                - "dxf": Ending 'dx' value          default: 0
                - "dko": Starting 'dk' value        default: 0
                - "dkw": Waypoint 'dk' value        default: 0
                - "dkf": Ending 'dk' value          default: 0
            - t_nsteps: int             default: 100
                Number of time steps for generating discrete values

            Returns:
            - x_vals: np.ndarray        shape: N,
                The 'x(t)' values
            - y_vals: np.ndarray        shape: N,
                The 'y(t)' values
            - th_vals: np.ndarray       shape: N,
                The 'theta(t)' values
            - t_vals: np.ndarray        shape: N,
                The corresponding time stamps
            - x_ft: sp.Expr     SymPy expression (`t` needed)
            - y_ft: sp.Expr     SymPy expression (`t` needed)
            - th_ft: sp.Expr    SymPy expression (`t` needed)
        """
        # Local names (avoid "self.")
        bb_tval, dbb_tval = self.bb_tval, self.dbb_tval
        t, t_f, t_o, t_w = self.t, self.t_f, self.t_o, self.t_w
        wx, wk = self.wx, self.wk
        mx, mdx, mk, mdk = self.mx, self.mdx, self.mk, self.mdk
        xo, xw, xf = self.xo, self.xw, self.xf
        dxo, dxw, dxf = self.dxo, self.dxw, self.dxf
        yo, yw, yf = self.yo, self.yw, self.yf
        tho, thw, thf = self.tho, self.thw, self.thf
        ko, kw, kf = self.ko, self.kw, self.kf
        # Substitution values for sympy symbols
        val_subs = {
            # Time values
            t_o: constrs.get('to', 0), t_w: constrs.get('tw', 20),
            t_f: constrs.get('tf', 50),
            # X values
            xo: constrs.get('xo', 0), xw: constrs.get('xw', 5), 
            xf: constrs.get('xf', 40),
            # Y values
            yo: constrs.get('yo', 0), yw: constrs.get('yw', 30), 
            yf: constrs.get('yf', 40),
            # k = tan(theta) values
            ko: constrs.get('ko', np.tan(0)), 
            kw: constrs.get('kw', np.tan(np.pi/5)), 
            kf: constrs.get('kf', np.tan(0)), 
            # x-dot constraints
            dxo: constrs.get('dxo', 0), 
            dxf: constrs.get('dxf', 0), 
            dxw: constrs.get('dxw', 0), 
            # k-dot constraints
            "dko": constrs.get('dko', 0), 
            "dkw": constrs.get('dkw', 0), 
            "dkf": constrs.get('dkf', 0),
        }
        # ===== Equations for 'wx_i' values =====
        mx_equations = [    # Constraints for 'x'
            sp.Eq(xo, mx.subs({t: t_o}).subs(val_subs)), 
            sp.Eq(xw, mx.subs({t: t_w}).subs(val_subs)), 
            sp.Eq(xf, mx.subs({t: t_f}).subs(val_subs)),
            sp.Eq(dxo, mdx.subs({t: t_o}).subs(val_subs)),
            sp.Eq(dxw, mdx.subs({t: t_w}).subs(val_subs)),
            sp.Eq(dxf, mdx.subs({t: t_f}).subs(val_subs)),
        ]
        # Solve for 'wx' values
        wx_sols = sp.solve(mx_equations, wx)
        wx_subs = dict()    # Solution (substituted) as a dict
        for i in range(self.n + 1):
            wx_sol_i = sp.simplify(wx_sols[wx[i]]).subs(val_subs)
            wx_subs[wx[i]] = wx_sol_i
        # ===== Equations for 'wk_i' values =====
        # Model 'y'
        mdxk = mdx.subs(wx_subs) * mk   # Inside the integral for y
        mdxk = mdxk.subs(val_subs)
        y_t = val_subs[yo] + sp.integrate(mdxk, 
            (t, val_subs[t_o], t))  # f(wk, t)
        # Make the rest a func. of (wk, t)
        k_t = mk.subs(val_subs) # k = tan(theta)
        dk_t = mdk.subs(val_subs)   # dk = w(1+sec^2(th))
        mk_equations = [    # Constraints for 'y', 'k' and 'dk'
            sp.Eq(y_t.subs({t: val_subs[t_o]}), val_subs[yo]),
            sp.Eq(y_t.subs({t: val_subs[t_w]}), val_subs[yw]),
            sp.Eq(y_t.subs({t: val_subs[t_f]}), val_subs[yf]),
            sp.Eq(k_t.subs({t: val_subs[t_o]}), val_subs[ko]),
            # sp.Eq(k_t.subs({t: val_subs[t_w]}), val_subs[kw]),
            sp.Eq(k_t.subs({t: val_subs[t_f]}), val_subs[kf]),
            sp.Eq(dk_t.subs({t: val_subs[t_o]}), val_subs["dko"]),
            # sp.Eq(dk_t.subs({t: val_subs[t_w]}), val_subs["dkw"]),
            sp.Eq(dk_t.subs({t: val_subs[t_f]}), val_subs["dkf"]),
        ]
        # Solve for 'wk' values
        wk_subs = sp.solve(mk_equations, wk)
        # ===== Results for the x, y, theta =====
        x_ft = mx.subs(wx_subs).subs(val_subs)
        y_ft = y_t.subs(wk_subs)
        k_ft = k_t.subs(wk_subs)
        th_ft = sp.atan(k_ft)
        # All symbols of time, convert to values (graph)
        t_vals = np.linspace(val_subs[t_o], val_subs[t_f], t_nsteps)
        x_vals = np.array(list(map(float, [
            x_ft.subs({t: tv}) for tv in t_vals])))
        y_vals = np.array(list(map(float, [
            y_ft.subs({t: tv}) for tv in t_vals])))
        th_vals = np.array(list(map(float, [
            th_ft.subs({t: tv}) for tv in t_vals])))
        # ===== Return values =====
        return x_vals, y_vals, th_vals, t_vals, x_ft, y_ft, th_ft


# %% See if object possible
if __name__ == "__main__":
    traj_planner = NonHoloThreePtBernstein()
    x_vals, y_vals, th_vals, t_vals, _, _, _ = \
        traj_planner.solve_wpt_constr()
    # Show output
    plt.figure()
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

# %% Experimental section

# %%
