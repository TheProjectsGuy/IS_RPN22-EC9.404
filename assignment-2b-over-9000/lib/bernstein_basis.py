# Interface for Bernstein Basis values
r"""
    Provides the Bernstein basis values for using with Bernstein
    polynomials [1][2]. The library extensively uses Sympy [3] for
    formulation and symbol substitution [4] will be needed to get
    exact values.

    [1]: https://en.wikipedia.org/wiki/Bernstein_polynomial
    [2]: https://www.particleincell.com/2012/bezier-splines/
    [3]: https://docs.sympy.org/latest/index.html
    [4]: https://docs.sympy.org/latest/modules/core.html#sympy.core.function.Subs
"""

# %% Import everything
import sympy as sp
import numpy as np
# If this script runs as main
from matplotlib import pyplot as plt

# %% Functions
# Get the basis terms for 'n'
def get_bernstein_basis(n_val = 5):
    r"""
        Returns the Bernstein basis polynomials (terms) for the
        Bernstein polynomials.

        $$
        b_{v,n}(x) = \;^n\textup{C}_v \; x^v \; \left ( 1 - x \right )^{(n-v)}
        $$

        The symbol of $x$ needs to be created (for substituting the 
        returned functions). Note that $x \in [0, 1]$ (the domain and
        range of Bernstein basis polynomials is [0, 1])

        Parameters:
        - n_val: int        default: 5      The value of $n$

        Returns:
        - bcoeffs_x_v: list[sp.Expr]     len = n+1
            The element $i$ is $b_{v=i,n=n_val}(x)$ sympy expression
    """
    x, v, n = sp.symbols("x, v, n")
    # Bernstein basis for v, n (func. of x)
    b_v_n = sp.binomial(n, v)*(x**v)*((1-x)**(n-v))
    bcoeffs_x_v: list[sp.Expr] = []    # Bernsterin Polynomials
    # For all 'v' value
    for i in range(n_val+1):
        b_nval = b_v_n.subs({v:i, n:n_val})
        b_term = sp.simplify(b_nval)
        # print(f"b(x;v={i},n={n_val}) = {b_term}")
        bcoeffs_x_v.append(b_term)
    return bcoeffs_x_v


# Time scaled basis and derivatives
def get_ts_bernstein_basis_diff(n_val = 5):
    r"""
        Returns the 'time scaled' bernstein basis and the derivatives.
        The time scaling is basically $\mu(t) = \sfrac{t-to}{to-tf}$,
        which maps $t$ to the range [0, 1].

        The following symbols need to be created to use the returned
        functions
        - $t$: The time value
        - $t_o$: The starting time. Also written as $to$
        - $t_f$: The ending time. Also written as $tf$

        This function gets the basis terms from `get_bernstein_basis`,
        does the time substitution (with $\mu$), calculates the time
        derivatives, and returns the basis and their derivatives.

        Note that the bernstein basis is given by

        $$
        b_{v,n}(x) = \;^n\textup{C}_v \; x^v \; \left ( 1 - x \right )^{(n-v)}
        $$

        Parameters:
        - n_val: int        default: 5      The value of $n$ above

        Returns:
        - bbterms_t: list[sp.Expr]      len: n+1
            The element $i$ is $b_{v=i,n=n_val}(\mu_t)$ sympy 
            expression. Note the time normalization.
            $\mu_t = \mu(t; to, tf) = \frac{t-to}{to-tf}$

        -dbbterms_t: list[sp.Expr]      len: n+1
            The derivative of the corresponding `bbterms_t[i]` w.r.t.
            time $t$
    """
    # Get the basis (in terms of 'x')
    basis = get_bernstein_basis(n_val)
    x = sp.symbols("x")
    # Time substitution
    t, to, tf = sp.symbols("t, t_o, t_f")   # Time, start, final
    mu = (t-to)/(tf-to) # Scale 't' to [0, 1]
    bbterms_t: list[sp.Expr] = [] # Basis in terms of 't'
    dbbterms_t: list[sp.Expr] = [] # Time derivatives
    for b in basis:
        # Basis substitution
        bbterm_mu = sp.simplify(b.subs({x:mu}))
        bbterms_t.append(bbterm_mu)
        # Differentiate basis term w.r.t. 't'
        dbbterm_t = sp.simplify(bbterm_mu.diff(t))
        dbbterms_t.append(dbbterm_t)
    # Return basis and time derivatives
    return bbterms_t, dbbterms_t


# %% Main
# Entrypoint
if __name__ == "__main__":
    n_val = 5
    # %% 
    # Bernstein basis
    basis = get_bernstein_basis(n_val)
    x = sp.symbols("x")
    print("Found the Bernstein Basis")
    for i, b in enumerate(basis):
        print(f"\tb(x;v={i},n={n_val}) = {b}")
    # %% 
    # Plot how the Bernstein polynomials evolve
    plt.figure(figsize=(6.4, 5), dpi=200)
    plt.title("Bernstein Basis Polynomials")
    x_values = np.linspace(0, 1, 50)
    total_vals = np.zeros_like(x_values)
    b: sp.Expr
    for i, b in enumerate(basis):
        # Visualize the plots
        by_vals = np.array([b.subs({x: xv}) for xv in x_values],
            float)
        total_vals += by_vals
        plt.plot(x_values, by_vals, label=f"$b_{{{i}, {n_val}}}$", 
            zorder=1)
        plt.axvline(i/n_val, c='gray', ls='--', lw=0.7, zorder=0)
    plt.plot(x_values, total_vals, ls='--', zorder=1, 
        label=f"$\sum_{{v=0}}^{{{n_val}}} b_{{v, {n_val}}}$")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.xlabel("x")
    plt.show()
    # %%
    t, to, tf = sp.symbols("t, t_o, t_f")   # Time, start, final
    mu = (t-to)/(tf-to) # Scale to [0, 1]
    bbterms_mu = [] # Bernstein basis terms in terms of 'mu'
    print("Substituting x with mu")
    for i, b in enumerate(basis):
        bbterm_mu = sp.simplify(b.subs({x:mu}))
        print(f"\tb(mu;v={i},n={n_val}) = {bbterm_mu}")
        bbterms_mu.append(bbterm_mu)
    # %% Manually differentiate for sanity check
    print("Differentiating the basis terms (time derivatives)")
    dbbterms_t = [] # Time derivates (d bb / dt for all)
    for i, b in enumerate(bbterms_mu):
        dbbterm_t = sp.simplify(b.diff(t))  # Time derivative
        print(f"\tdb(mu;v={i},n={n_val}) = {dbbterm_t}")
        dbbterms_t.append(dbbterm_t)
    # %% 
    # Test basis and time derivatives
    fbbterms_t, fdbbterms_t = get_ts_bernstein_basis_diff(n_val)
    # %%
    # Same Bernstein basis
    a = all([sp.simplify(bbterms_mu[i]-fbbterms_t[i])==0 \
        for i in range(6)])
    # Same Time derivatives
    b = all([sp.simplify(dbbterms_t[i]-fdbbterms_t[i])==0 \
        for i in range(6)])
    if a and b:
        print(f"The functions seem to be working")
    else:
        print(f"Some function is not working: basis:{a}, deriv.:{b}")
    pass

# %% Experimental section

# %%
