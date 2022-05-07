# Test the starter code and library compatibility
"""
    The provided starter code is compared with my implementation 
    (custom).

    - Bernstein Basis Polynomials: SUCCESS
    - Derivative of Bernstein basis elements: SUCCESS
    - Product functions: FAILED
        - The starter code given works for only one specific set of
            constraints and cannot be generalized. This requires us
            to frame the equations and use Wolfram Alpha to perform
            the integrals.
        - The custom implementation does around this using sympy 
            integration.
"""

# %% Import everything
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
# Both codes
from src.starter_code import get_bernstein_poly, \
    get_bernstein_differentials, get_product_functions
from src.bernstein_basis import get_ts_bernstein_basis_diff
from src.three_point_traj_planner import NonHoloThreePtBernstein


# %% Functions
def get_plots_starter_code(constrs: dict = dict()):
    n_val = 5
    # Get Bernstein basis and derivative
    bb_tval, dbb_tval = get_ts_bernstein_basis_diff(n_val)
    t, t_f, t_o, t_w = sp.symbols("t, t_f, t_o, t_w")
    # Weights
    wx = [sp.symbols(f"w_{{x_{i}}}") for i in range(n_val+1)]   # X
    wk = [sp.symbols(f"w_{{k_{i}}}") for i in range(n_val+1)]   # K
    # Build models
    mx: sp.Expr = 0     # Model for x
    mdx: sp.Expr = 0    # Model for dx
    mk: sp.Expr = 0     # Model for k = tan(theta)
    mdk: sp.Expr = 0    # Model for dk = w(1 + sec^2 (theta))
    for i in range(n_val+1):
        mx += wx[i] * bb_tval[i]    # For 'x'
        mdx += wx[i] * dbb_tval[i]  # For 'dx'
        mk += wk[i] * bb_tval[i]    # For 'k'
        mdk += wk[i] * dbb_tval[i]  # For 'dk'
    # X value of path - Start, waypoint, final
    xo, xw, xf = sp.symbols(r"x_o, x_w, x_f")
    dxo, dxw, dxf = sp.symbols(r"\dot{x}_o, \dot{x}_w, \dot{x}_f")
    # Y value of path
    yo, yw, yf = sp.symbols(r"y_o, y_w, y_f")
    # Theta (heading) of path (in rad.)
    tho, thw, thf = sp.symbols(r"\theta_o, \theta_w, \theta_f")
    ko, kw, kf = map(sp.tan, (tho, thw, thf))   # k = tan(theta)
    # val_subs = {
    #     t_o: 0, t_w: 10, t_f: 50,
    #     xo: 0, xw: 5, xf: 40,
    #     yo: 0, yw: 40, yf: 40,
    #     ko: np.tan(0), kf: np.tan(0), kw: np.tan(np.pi/4),
    #     dxo: 0, dxf: 0, dxw: 0,
    #     "dko": 0, "dkw": 0, "dkf": 0
    # }
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
        dxw: constrs.get('dxw', 0), 
        dxf: constrs.get('dxf', 0), 
        # k-dot constraints
        "dko": constrs.get('dko', 0), 
        "dkw": constrs.get('dkw', 0), 
        "dkf": constrs.get('dkf', 0),
    }
    # x(t) equation constraints
    mx_equations = [
        sp.Eq(xo, mx.subs({t: t_o}).subs(val_subs)), 
        sp.Eq(xw, mx.subs({t: t_w}).subs(val_subs)), 
        sp.Eq(xf, mx.subs({t: t_f}).subs(val_subs)),
        sp.Eq(dxo, mdx.subs({t: t_o}).subs(val_subs)),
        sp.Eq(dxw, mdx.subs({t: t_w}).subs(val_subs)),
        sp.Eq(dxf, mdx.subs({t: t_f}).subs(val_subs)),
    ]
    # Solve for 'wx' values
    wx_sols = sp.solve(mx_equations, wx)
    wx_subs = dict()
    for i in range(n_val + 1):
        wx_sol_i = sp.simplify(wx_sols[wx[i]]).subs(val_subs)
        wx_subs[wx[i]] = wx_sol_i
    mdxk = mdx.subs(wx_subs) * mk   # Inside the integral for y
    mdxk = mdxk.subs(val_subs)
    y_t = val_subs[yo] + sp.integrate(mdxk, (t, val_subs[t_o], t))
    k_t = mk.subs(val_subs)
    dk_t = mdk.subs(val_subs)
    # Get the product functions
    wx0, wx1, wx2, wx3, wx4, wx5 = map(float, [wx_subs[wx[i]] \
        for i in range(n_val+1)])
    t0_val, tf_val = val_subs[t_o], val_subs[t_f]
    x_vals = []
    y_vals = []
    th_vals = []
    for tv in t_vals:
        # 'wk' values
        wk0, wk1, wk2, wk3, wk4, wk5 = get_product_functions(wx0, 
        wx1, wx2, wx3, wx4, wx5, t0_val, tv, tf_val)
        x_vals.append(float(mx.subs({wx[0]: wx0, wx[1]: wx1, 
            wx[2]: wx2, wx[3]: wx3, wx[4]: wx4, wx[5]: wx5, 
            t_f: tf_val, t_o: t0_val, t: tv})))
        y_vals.append(float(y_t.subs({wk[0]: wk0, wk[1]: wk1, 
            wk[2]: wk2, wk[3]: wk3, wk[4]: wk4, wk[5]: wk5, t: tv})))
        k_val = float(y_t.subs({wk[0]: wk0, wk[1]: wk1, wk[2]: wk2,
            wk[3]: wk3, wk[4]: wk4, wk[5]: wk5, t: tv, t_o: t0_val, 
            t_f: tf_val}))
        th_vals.append(np.arctan(k_val))
    x_vals = np.array(x_vals, float)
    y_vals = np.array(y_vals, float)
    th_vals = np.array(th_vals, float)
    return x_vals, y_vals, th_vals, t_vals


# %%
if __name__ == "__main__":
    # Parameters for testing
    to, tf = 0, 10       # Starting and ending time
    t = np.linspace(to, tf, 50) # All timestamps to test

    # %%
    # ---- Verify Bernstein basis elements ----
    # For the source code provided
    sc_bb_tvals = [list(get_bernstein_poly(to, tv, tf)) for tv in t]
    sc_bb_tvals = np.array(sc_bb_tvals)
    # For the custom implementation
    n_val = 5
    bbterms_t, dbbterms_t = get_ts_bernstein_basis_diff(n_val)
    to_val, t_vals, tf_val = to, t, tf  # Store values
    to_sp, t_sp, tf_sp = sp.symbols("t_o, t, t_f")   # Symbols
    subs_bbterms_t = lambda to_val, tv, tf_val: list(map(float, 
        [bbterms_t[i].subs({to_sp: to_val, tf_sp: tf_val, t_sp:tv}) \
            for i in range(0, n_val+1)]))   # Basis Substitution
    ud_bb_tvals = [subs_bbterms_t(to_val, tv, tf_val) \
        for tv in t_vals
    ]
    ud_bb_tvals = np.array(ud_bb_tvals)
    # Result
    if np.allclose(ud_bb_tvals, sc_bb_tvals):
        print(f"[SUCCESS]: Bernstein basis values are matching")
    else:
        print(f"[ERROR]: Bernstein basis values do not match")

    # %%
    # ---- Verify Derivative of Bernstein basis elements ----
    # For source code provided
    sc_dbb_tvals = [get_bernstein_differentials(to, tv, tf) \
        for tv in t]
    sc_dbb_tvals = np.array(sc_dbb_tvals)
    # For the custom implementation
    subs_dbbterms_t = lambda to_val, tv, tf_val: list(map(float,[
        dbbterms_t[i].subs({
            to_sp: to_val, tf_sp: tf_val, t_sp: tv}) \
                for i in range(0, n_val+1)]))   # Subs diff. basis
    ud_dbb_tvals = [subs_dbbterms_t(to_val, tv, tf_val) \
        for tv in t_vals
    ]
    ud_dbb_tvals = np.array(ud_dbb_tvals)
    # Result
    if np.allclose(ud_bb_tvals, sc_bb_tvals):
        print(f"[SUCCESS]: Derivative of Bernstein basis values are "
                f"matching")
    else:
        print(f"[ERROR]: Derivative of Bernstein basis values do not "
                f"match")

    # %%
    # ---- Verify the product function output ----
    # constrs = {
    #     "to": 0, "tw": 10, "tf": 50,
    #     "xo": 0, "xw": 5, "xf": 40,
    #     "yo": 0, "yw": 40, "yf": 40,
    #     "ko": np.tan(0), "kw": np.tan(np.pi/4), "kf": np.tan(0), 
    #     "dxo": 0, "dxw": 0, "dxf": 0,
    #     "dko": 0, "dkw": 0, "dkf": 0
    # }
    constrs = {
        "to": 0, "tw": 20, "tf": 50,
        "xo": 0, "xw": 5, "xf": 40,
        "yo": 0, "yw": 30, "yf": 40,
        "ko": np.tan(0), "kw": np.tan(np.pi/5), "kf": np.tan(0), 
        "dxo": 0, "dxw": 0, "dxf": 0,
        "dko": 0, "dkw": 0, "dkf": 0
    }
    # -- Custom implementation --
    traj_planner = NonHoloThreePtBernstein()
    x_vals, y_vals, th_vals, t_vals, _, _, _ = \
        traj_planner.solve_wpt_constr(constrs)
    # Show output
    plt.figure(figsize=(4, 9))
    plt.suptitle("Custom Implementation")
    plt.subplot(3, 1, 1)
    plt.title("X plot")
    plt.plot(t_vals, x_vals, 'r')
    plt.subplot(3, 1, 2)
    plt.title("Y plot")
    plt.plot(t_vals, y_vals, 'g')
    plt.subplot(3, 1, 3)
    plt.title("Theta plot")
    plt.plot(t_vals, th_vals, 'b')
    plt.show(block=False)
    # X Y plot
    plt.figure()
    plt.suptitle("Custom Implementation")
    plt.title("XY time-plot")
    plt.scatter(x_vals, y_vals, 1.0, c=t_vals)
    plt.colorbar()
    plt.show(block=False)
    # -- Starter code implementation --
    x_vals, y_vals, th_vals, t_vals = get_plots_starter_code(constrs)
    # Show output
    plt.figure(figsize=(4, 9))
    plt.suptitle("Starter Code")
    plt.subplot(3, 1, 1)
    plt.title("X plot")
    plt.plot(t_vals, x_vals, 'r')
    plt.subplot(3, 1, 2)
    plt.title("Y plot")
    plt.plot(t_vals, y_vals, 'g')
    plt.subplot(3, 1, 3)
    plt.title("Theta plot")
    plt.plot(t_vals, th_vals, 'b')
    plt.show(block=False)
    # X Y plot
    plt.figure()
    plt.suptitle("Starter Code")
    plt.title("XY time-plot")
    plt.scatter(x_vals, y_vals, 1.0, c=t_vals)
    plt.colorbar()
    plt.show()


# %% Experimental section

# %%
