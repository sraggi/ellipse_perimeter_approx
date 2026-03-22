

import numpy as np
from scipy.special import ellipe
from scipy.optimize import differential_evolution, minimize


# ------------------------------------------------------------
# Compute the parameter
# h = ((a-b)/(a+b))^2
# This parameter appears in many approximations of the
# perimeter of an ellipse.
# ------------------------------------------------------------
def h_of(a, b):
    return ((a - b) / (a + b)) ** 2


# ------------------------------------------------------------
# Ramanujan II approximation of the ellipse perimeter.
#
# PR2(a,b) = π(a+b) * ( 1 + 3h / (10 + sqrt(4-3h)) )
#
# This is already a very accurate approximation and will be
# used as the numerator of the corrected formula.
# ------------------------------------------------------------
def pr2_ramanujan(a, b):
    h = h_of(a, b)
    return np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h)))


# ------------------------------------------------------------
# Exact ellipse perimeter using the complete elliptic integral
# of the second kind.
#
# P = 4a E(m)
#
# where
# m = 1 - (b/a)^2
#
# scipy.special.ellipe(m) computes E(m).
# ------------------------------------------------------------
def perimeter_exact(a, b):
    # a>=b here (a from 1 to 100, b=1)
    m = 1.0 - (b / a) ** 2
    return 4.0 * a * ellipe(m)


# ------------------------------------------------------------
# Main function that performs the minimax fitting.
#
# The goal is to determine constants A and B in the formula
#
#     Pcorr = PR2 / (1 - A exp(-B(1-h)))
#
# by minimizing the maximum relative error over a grid of
# ellipse shapes.
# ------------------------------------------------------------
def fit_minimax_AB(n_grid=3000, a_min=1.0, a_max=100.0, b=1.0, seed=123):

    # --------------------------------------------------------
    # Create the grid of ellipse shapes.
    # Here a varies from a_min to a_max using n_grid points.
    # --------------------------------------------------------
    a = np.linspace(a_min, a_max, n_grid, dtype=np.float64)

    # --------------------------------------------------------
    # Precompute values that do not depend on A and B.
    # This greatly accelerates the optimization process.
    # --------------------------------------------------------
    Pex = perimeter_exact(a, b)
    PR2 = pr2_ramanujan(a, b)
    t = 1.0 - h_of(a, b)  # t = 1 - h

    # --------------------------------------------------------
    # Objective function for the minimax optimization.
    #
    # The optimizer searches for the parameters A and B that
    # minimize the maximum relative error over the grid.
    # --------------------------------------------------------
    def obj(x):

        # Extract parameters
        A, B = float(x[0]), float(x[1])

        # Compute denominator of corrected approximation
        denom = 1.0 - A * np.exp(-B * t)

        # Avoid invalid values in the denominator
        if np.any(denom <= 0.0):
            return 1e9

        # Compute corrected approximation
        Papp = PR2 / denom

        # Relative error
        rel = np.abs(Papp - Pex) / Pex

        # Minimax criterion: return the largest error
        return float(np.max(rel))

    # --------------------------------------------------------
    # Global optimization stage.
    #
    # differential_evolution is a robust global optimizer
    # that helps avoid local minima.
    # --------------------------------------------------------
    bounds = [(0.0, 2.0e-3), (0.1, 60.0)]

    de = differential_evolution(
        obj,
        bounds=bounds,
        seed=seed,
        popsize=18,
        tol=1e-10,
        mutation=(0.5, 1.0),
        recombination=0.7,
        updating="deferred",
        polish=False,
        workers=1,
    )

    # --------------------------------------------------------
    # Local refinement stage.
    #
    # Powell's method is used to refine the solution obtained
    # by the global search.
    # --------------------------------------------------------
    loc = minimize(
        obj,
        x0=de.x,
        method="Powell",
        bounds=bounds,
        options=dict(maxiter=2000, xtol=1e-14, ftol=1e-14),
    )

    # Extract final parameters
    A, B = map(float, loc.x)

    # Maximum relative error
    f = float(loc.fun)

    # --------------------------------------------------------
    # Determine where the worst error occurs in the grid.
    # --------------------------------------------------------
    denom = 1.0 - A * np.exp(-B * t)
    rel = np.abs((PR2 / denom) - Pex) / Pex

    # Index of maximum error
    i = int(np.argmax(rel))

    # Return results
    return {
        "A": A,
        "B": B,
        "max_rel_error": f,
        "worst_a": float(a[i]),
        "n_grid": n_grid,
        "success_global": bool(de.success),
        "success_local": bool(loc.success),
    }


# ------------------------------------------------------------
# Program execution
# ------------------------------------------------------------
if __name__ == "__main__":

    # Run the fitting procedure
    out = fit_minimax_AB(n_grid=3000, a_min=1.0, a_max=100.0, b=1.0, seed=123)

    # Print results
    print("=== Minimax fit (b=1, a in [1,100], 3000 points) ===")
    print(f"A = {out['A']:.10e}")
    print(f"B = {out['B']:.6f}")
    print(f"max relative error = {out['max_rel_error']:.6e}")
    