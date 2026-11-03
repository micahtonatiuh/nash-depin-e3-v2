"""
tape_core.py — Random tape and simulation reference for the R1.4 physical test
(Paper 1 revision, Access-2026-41887). Nash-DePIN E5.

Does NOT modify the lab code: it imports encode_state and apply_actions_logic
from experiments/run_e5.py and joint_nash from nash_depin/nash_solver.py.

Idea
----
The stochastic policy (sample sigma_i, with exploration epsilon) produces task
hand-overs that the deterministic policy never shows. To compare the physical
run against a UNIQUE simulation reference, all randomness comes from a
pre-recorded tape: two uniforms per robot and step,
    u_explore[t, i]  -> explore if u_explore < epsilon
    u_pick[t, i]     -> exploratory action  floor(u_pick * n_actions)
                        policy action        inverse CDF of sigma_i at u_pick
The tape depends only on (seed, episode), never on the state. If the robots
end each step in the commanded cell, the physical run sees the same states as
the simulation, so it takes exactly the same actions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from experiments.run_e5 import encode_state, apply_actions_logic  # noqa: E402
from nash_depin.nash_solver import joint_nash  # noqa: E402

N_ACTIONS = 5
TAPE_VERSION = 1


# ------------------------------------------------------------------ #
# Tape                                                                #
# ------------------------------------------------------------------ #

def make_tape(seed: int, episode: int, steps: int) -> np.ndarray:
    """Uniforms in [0,1), shape (steps, 2 robots, 2 [explore, pick])."""
    rng = np.random.default_rng(np.random.SeedSequence([20261010, seed, episode]))
    return rng.random((steps, 2, 2))


def tape_action(sigma: np.ndarray, u_explore: float, u_pick: float,
                epsilon: float, greedy: bool) -> int:
    """Action from the tape. greedy=True reproduces the E3 policy (argmax)."""
    n = len(sigma)
    if u_explore < epsilon:
        return min(int(u_pick * n), n - 1)
    if greedy:
        return int(np.argmax(sigma))
    p = np.clip(np.asarray(sigma, float), 0.0, None)
    cdf = np.cumsum(p / p.sum())
    return min(int(np.searchsorted(cdf, u_pick, side="right")), n - 1)


# ------------------------------------------------------------------ #
# Policy                                                              #
# ------------------------------------------------------------------ #

def load_q(path) -> np.ndarray:
    return np.load(path)["q_table"]


def policy_sigmas(q0, q1, state, equity=True):
    """Same call as PhysicalNashTeam.act in run_e5.py (fast=True)."""
    s0, s1, _, _ = joint_nash(q0[state], q1[state], fast=True, equity=equity)
    return s0, s1


def decide(q0, q1, cells, grid, tape_row, epsilon, greedy, equity=True):
    state = encode_state(cells, grid)
    s0, s1 = policy_sigmas(q0, q1, state, equity)
    acts = tuple(tape_action(s, tape_row[i, 0], tape_row[i, 1], epsilon, greedy)
                 for i, s in enumerate((s0, s1)))
    return state, acts, (s0, s1)


# ------------------------------------------------------------------ #
# Simulation reference                                                #
# ------------------------------------------------------------------ #

def simulate_episode(q0, q1, env, grid, start_cells, tape, epsilon, greedy,
                     stale_cut=40, equity=True):
    """One episode with exactly the step logic of run_e5.py.

    stale_cut reproduces run_e5's detector (cut after `stale_cut` steps with no
    task completed); None disables it.
    """
    cells = [tuple(c) for c in start_cells]
    cum_r = [0.0, 0.0]
    cum_t = [0, 0]
    prev_t = [0, 0]
    stale = 0
    cols = 0
    coops = 0
    moves = [0, 0]
    steps = []
    for t in range(tape.shape[0]):
        state, acts, sig = decide(q0, q1, cells, grid, tape[t], epsilon, greedy, equity)
        new_cells, rewards, info, mover = apply_actions_logic(cells, acts, grid, env)
        steps.append({
            "step": t,
            "cells": [list(c) for c in cells],
            "act": list(acts),
            "new_cells": [list(c) for c in new_cells],
            "col": info["collisions"],
            "task": list(info["tasks"]),
            "sigma_of_act": [round(float(sig[i][acts[i]]), 6) for i in range(2)],
        })
        # run_e5 detector, same order: checked on the counters BEFORE this
        # step is applied; on a cut the step is logged but not executed.
        if cum_t == prev_t:
            stale += 1
        else:
            stale = 0
            prev_t = list(cum_t)
        if stale_cut is not None and stale >= stale_cut:
            steps[-1]["cut"] = True
            break
        for i in range(2):
            cum_r[i] += rewards[i]
            cum_t[i] += info["tasks"][i]
            moves[i] += int(mover[i])
        cols += info["collisions"]
        coops += info["cooperations"]
        cells = [tuple(c) for c in new_cells]
    return {
        "steps": steps,
        "n_steps": len(steps),
        "reward": [round(x, 1) for x in cum_r],
        "tasks": cum_t,
        "collisions": cols,
        "cooperations": coops,
        "moves": moves,
        "handovers": count_handovers(steps),
        "final_cells": [list(c) for c in cells],
    }


def count_handovers(steps) -> int:
    """Changes of the robot that completes the task (custody transfers)."""
    holder = None
    n = 0
    for s in steps:
        who = [i for i in range(2) if s["task"][i]]
        if who:
            if holder is not None and who[0] != holder:
                n += 1
            holder = who[0]
    return n


def gini2(r):
    r = np.asarray(r, float)
    tot = r.sum()
    return float(abs(r[0] - r[1]) / (2 * tot)) if tot > 0 else float("nan")


def first_divergence(ref_steps, phy_steps, key="cells"):
    """First step whose logical cells differ (None if identical prefix)."""
    n = min(len(ref_steps), len(phy_steps))
    for k in range(n):
        if ref_steps[k][key] != phy_steps[k][key]:
            return k
    return None


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1))
