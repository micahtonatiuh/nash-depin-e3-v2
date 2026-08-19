"""
nash_solver.py — Bimatrix Nash equilibrium solver with egalitarian refinement.
Nash-DePIN.

Byte-identical to nash_depin/nash_solver.py in the E2 repository: the physical
runs must use exactly the solver that produced the transferred Q-tables.

Implements Hu & Wellman (2003) Nash-Q: the stage-game equilibrium is computed
jointly over BOTH payoff matrices (A for agent 0, B for agent 1).

EQUILIBRIUM SELECTION REFINEMENT (equity=True)
----------------------------------------------
General-sum games over a scarce resource (such as MRTA with a single task)
admit MULTIPLE Nash equilibria. In competition for one task, "agent 0 always
takes it", "agent 1 always takes it", and "both take it with probability 1/2"
are all valid equilibria.

Without a selection criterion, learning converges to any one of them depending
on initial conditions — including monopoly equilibria, which are unfair (high
Gini). This produces bimodal outcomes.

We therefore apply an EGALITARIAN REFINEMENT: among the available Nash
equilibria, we select the one that MINIMIZES the payoff disparity |V0 - V1|.
This is a standard equilibrium-selection criterion in game theory (related to
the Nash 1950 bargaining solution and egalitarian/Rawlsian refinements). It
does NOT modify the equilibrium set — it only chooses, among equally valid
equilibria, the most equitable one. It is reported as part of the method.

Contrast with QMIX/VDN: those methods cannot select an equitable equilibrium
because their structure (shared reward + monotonicity/additivity) does not
represent the individual payoffs required to evaluate equity.

References:
  - Hu & Wellman (2003), Nash Q-Learning for General-Sum Stochastic Games
  - Nash (1950), The Bargaining Problem (egalitarian payoff solution)
"""

import numpy as np
from typing import Tuple, List

try:
    import nashpy as nash
    _HAS_NASHPY = True
except ImportError:
    _HAS_NASHPY = False


def _best_response_iterate(A, B, init0, init1, iters=60):
    """Smoothed best-response iteration from a given initialization."""
    n = A.shape[0]
    s0 = init0.copy()
    s1 = init1.copy()
    for _ in range(iters):
        br0 = np.zeros(n); br0[int(np.argmax(A @ s1))] = 1.0
        br1 = np.zeros(n); br1[int(np.argmax(B.T @ s0))] = 1.0
        s0 = 0.9 * s0 + 0.1 * br0
        s1 = 0.9 * s1 + 0.1 * br1
    V0 = float(s0 @ A @ s1)
    V1 = float(s0 @ B @ s1)
    return s0, s1, V0, V1


def _select(candidates, equity):
    """
    Select one equilibrium from the candidate list.

    equity=True  -> EGALITARIAN refinement: minimize |V0 - V1| (fairest).
    equity=False -> maximum social welfare: maximize V0 + V1 (efficiency).
    """
    if equity:
        return min(candidates, key=lambda c: abs(c[2] - c[3]))
    return max(candidates, key=lambda c: c[2] + c[3])


def joint_nash(A: np.ndarray, B: np.ndarray,
               fast: bool = False, equity: bool = False
               ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Nash equilibrium of the bimatrix game (A, B).

    A[i,j] = agent 0's payoff when agent 0 plays i, agent 1 plays j
    B[i,j] = agent 1's payoff for the same joint action

    Args:
        fast   : skip nashpy and use best-response iteration (used in E2/E5,
                 where the solver is called millions of times).
        equity : apply the EGALITARIAN equilibrium-selection refinement
                 (minimize |V0 - V1|). If False, maximize social welfare.

    Returns:
        sigma0, sigma1 : mixed strategies (sum to 1)
        V0, V1         : expected value of each agent at the equilibrium
    """
    n = A.shape[0]
    candidates: List[Tuple[np.ndarray, np.ndarray, float, float]] = []

    # --- Exact path: nashpy enumerates ALL equilibria ---
    if _HAS_NASHPY and not fast:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                game = nash.Game(A, B)
                eqs = list(game.support_enumeration())
            for s0, s1 in eqs:
                s0 = np.clip(np.asarray(s0, float), 0, 1)
                s1 = np.clip(np.asarray(s1, float), 0, 1)
                s0 /= s0.sum() + 1e-12
                s1 /= s1.sum() + 1e-12
                V0 = float(s0 @ A @ s1)
                V1 = float(s0 @ B @ s1)
                candidates.append((s0, s1, V0, V1))
        except Exception:
            pass

    # --- Fast path: best-response from MULTIPLE initializations ---
    # With equity=True several initializations are used so that monopoly
    # equilibria on BOTH sides are discovered; the equitable equilibrium then
    # also appears among the candidates and can be picked by _select.
    if not candidates:
        unif = np.ones(n) / n
        starts = [(unif, unif)]
        if equity:
            e0 = np.zeros(n); e0[0] = 1.0
            e1 = np.zeros(n); e1[min(1, n - 1)] = 1.0
            starts += [(e0, unif), (unif, e0), (e0, e1)]
        for init0, init1 in starts:
            candidates.append(_best_response_iterate(A, B, init0, init1))

    return _select(candidates, equity)


def nash_gap_joint(A: np.ndarray, B: np.ndarray, equity: bool = False) -> float:
    """
    Nash Gap (exploitability) of the bimatrix game (A, B).

    gap -> 0 iff exact equilibrium. Uses the same equilibrium as joint_nash
    (with the egalitarian refinement when equity=True).
    """
    sigma0, sigma1, V0, V1 = joint_nash(A, B, equity=equity)
    best_dev_0 = float(np.max(A @ sigma1))
    best_dev_1 = float(np.max(B.T @ sigma0))
    gap0 = max(0.0, best_dev_0 - V0)
    gap1 = max(0.0, best_dev_1 - V1)
    return max(gap0, gap1)
