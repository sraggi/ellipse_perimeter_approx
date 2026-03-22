# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:35:39 2026

@author: saraggi
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# scipy provides fast numerical tools
from scipy.special import ellipe
from scipy.optimize import differential_evolution, minimize


# ============================================================
# Mathematical functions
# ============================================================

def h_of(a: np.ndarray, b: float) -> np.ndarray:
    """
    Compute the parameter

        h = ((a - b)/(a + b))^2

    used in many approximations of the ellipse perimeter.

    The function works with numpy arrays so that
    all values can be computed at once (vectorized).
    """
    return ((a - b) / (a + b)) ** 2


def pr2_ramanujan(a: np.ndarray, b: float) -> np.ndarray:
    """
    Ramanujan II approximation of the ellipse perimeter.

    PR2(a,b) = pi*(a+b) * (1 + 3h/(10 + sqrt(4 - 3h)))

    This approximation is already extremely accurate,
    and we will use it as the numerator of the new formula.
    """
    h = h_of(a, b)

    return np.pi * (a + b) * (
        1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h))
    )


def perimeter_exact(a: np.ndarray, b: float) -> np.ndarray:
    """
    Exact ellipse perimeter computed using the
    complete elliptic integral of the second kind.

        P = 4a * E(m)

    where

        m = 1 - (b/a)^2

    scipy.special.ellipe(m) evaluates E(m).

    This function is vectorized, so it computes
    many values simultaneously and is very fast.
    """

    m = 1.0 - (b / a) ** 2

    return 4.0 * a * ellipe(m)


# ============================================================
# Configuration parameters
# ============================================================

@dataclass
class FitConfig2Exp:
    """
    This class stores all parameters used in the fitting process.
    """

    # fixed semi-minor axis
    b: float = 1.0

    # range for the semi-major axis
    a_min: float = 1.0
    a_max: float = 1000.0

    # number of points in the numerical grid
    n_grid: int = 3000

    # random seed for reproducibility
    seed: int = 123

    # constraint A + C = S
    S: float = 4.023374941e-4

    # search ranges for parameters
    B_bounds: Tuple[float, float] = (0.1, 80.0)
    D_bounds: Tuple[float, float] = (0.1, 200.0)


# ============================================================
# Main minimax fitting procedure
# ============================================================

def fit_minimax_2exp(cfg: FitConfig2Exp) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Step 1: build the grid of ellipse shapes
    # --------------------------------------------------------

    # a varies from 1 to 1000 using 3000 points
    a = np.linspace(cfg.a_min, cfg.a_max, cfg.n_grid)

    # --------------------------------------------------------
    # Step 2: precompute quantities that do NOT depend on A,B,C,D
    # --------------------------------------------------------

    # exact perimeter
    P_exact = perimeter_exact(a, cfg.b)

    # Ramanujan II approximation
    PR2 = pr2_ramanujan(a, cfg.b)

    # t = 1 - h appears in the exponentials
    t = 1.0 - h_of(a, cfg.b)

    # --------------------------------------------------------
    # Step 3: parameter unpacking
    # --------------------------------------------------------

    def unpack(x):
        """
        The optimizer will search only over (B, C, D).

        Since we impose the constraint

            A + C = S

        we compute

            A = S - C
        """

        B, C, D = x
        A = cfg.S - C

        return A, B, C, D


    # --------------------------------------------------------
    # Step 4: minimax objective function
    # --------------------------------------------------------

    def objective(x):
        """
        Compute the maximum relative error over the grid.

        The optimizer will try to MINIMIZE this value.
        """

        A, B, C, D = unpack(x)

        # safety checks
        if A < 0 or C < 0 or B <= 0 or D <= 0:
            return 1e50

        # corrected approximation
        denom = 1.0 - (
            A * np.exp(-B * t) +
            C * np.exp(-D * t)
        )

        # avoid invalid denominators
        if np.any(denom <= 0):
            return 1e50

        P_approx = PR2 / denom

        # relative error
        rel_error = np.abs(P_approx - P_exact) / P_exact

        # minimax criterion
        return float(np.max(rel_error))


    # --------------------------------------------------------
    # Step 5: global optimization
    # --------------------------------------------------------

    bounds = [
        cfg.B_bounds,     # B
        (0.0, cfg.S),     # C (so A = S - C >= 0)
        cfg.D_bounds      # D
    ]

    result_global = differential_evolution(
        objective,
        bounds=bounds,
        seed=cfg.seed,
        popsize=18,
        tol=1e-10,
        mutation=(0.5, 1.0),
        recombination=0.7,
        updating="deferred",
        polish=False
    )


    # --------------------------------------------------------
    # Step 6: local refinement
    # --------------------------------------------------------

    result_local = minimize(
        objective,
        x0=result_global.x,
        method="Powell",
        bounds=bounds,
        options=dict(
            maxiter=2500,
            xtol=1e-14,
            ftol=1e-14
        )
    )


    # --------------------------------------------------------
    # Step 7: final parameters
    # --------------------------------------------------------

    B, C, D = result_local.x
    A = cfg.S - C

    max_error = result_local.fun


    # --------------------------------------------------------
    # Step 8: locate worst-case ellipse
    # --------------------------------------------------------

    denom = 1.0 - (
        A * np.exp(-B * t) +
        C * np.exp(-D * t)
    )

    P_approx = PR2 / denom
    rel = np.abs(P_approx - P_exact) / P_exact

    idx = np.argmax(rel)

    return dict(
        A=A,
        B=B,
        C=C,
        D=D,
        max_relative_error=max_error,
        worst_case_a=float(a[idx])
    )


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":

    config = FitConfig2Exp()

    result = fit_minimax_2exp(config)

    print("\nOptimal parameters:")
    print(f"A = {result['A']:.12e}")
    print(f"B = {result['B']:.6f}")
    print(f"C = {result['C']:.12e}")
    print(f"D = {result['D']:.6f}")

    print("\nMaximum relative error:")
    print(result["max_relative_error"])

    