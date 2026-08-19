"""
test_e5.py — Tests for the physical-run logic (no ROS: environment logic,
state encoding, conflict resolution).
Run: pytest tests/test_e5.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Pure logic only (no rclpy needed — see run_e5.py module docstring)
from experiments.run_e5 import encode_state, apply_actions_logic

ENV = {"task_cell": [0, 0], "task_reward": 10.0, "collision_penalty": -5.0,
       "cooperation_reward": 2.0, "coop_distance": 1}


# --- State encoding: must be identical to E2 ---

def test_encode_matches_e2_formula():
    """((f0*5+c0)*25) + (f1*5+c1)"""
    assert encode_state([(0, 0), (0, 0)], 5) == 0
    assert encode_state([(0, 0), (4, 4)], 5) == 24
    assert encode_state([(1, 2), (3, 4)], 5) == (1 * 5 + 2) * 25 + (3 * 5 + 4)

def test_encode_range():
    assert encode_state([(4, 4), (4, 4)], 5) == 624  # last of the 625 states


# --- Action logic ---

def test_normal_move():
    cells = [(2, 2), (4, 4)]
    new, rw, info, mover = apply_actions_logic(cells, (3, 4), 5, ENV)  # right, stay
    assert new[0] == (2, 3) and new[1] == (4, 4)
    assert mover == [True, False]
    assert info["collisions"] == 0

def test_wall_clamp():
    """An action into a wall does not leave the grid."""
    new, _, _, _ = apply_actions_logic([(0, 0), (4, 4)], (0, 1), 5, ENV)  # up/down
    assert new[0] == (0, 0) and new[1] == (4, 4)

def test_same_cell_collision():
    """Both target the same cell: nobody moves, both get -5."""
    cells = [(2, 1), (2, 3)]
    new, rw, info, mover = apply_actions_logic(cells, (3, 2), 5, ENV)  # right, left -> (2,2)
    assert info["collisions"] == 1
    assert new == cells
    assert mover == [False, False]
    assert rw[0] == -5.0 and rw[1] == -5.0

def test_swap_collision():
    """A cell swap is also a collision (physical crossing)."""
    cells = [(2, 2), (2, 3)]
    new, rw, info, mover = apply_actions_logic(cells, (3, 2), 5, ENV)
    assert info["collisions"] == 1
    assert mover == [False, False]

def test_task_reward_at_station():
    """Reaching (0,0) gives +10 to exactly ONE robot."""
    cells = [(0, 1), (4, 4)]
    new, rw, info, _ = apply_actions_logic(cells, (2, 4), 5, ENV)  # left -> (0,0), stay
    assert new[0] == (0, 0)
    assert rw[0] >= 10.0
    assert info["tasks"] == [1, 0]

def test_camping_recollects():
    """Staying on the station re-collects (same semantics as the simulation)."""
    cells = [(0, 0), (4, 4)]
    _, rw, info, _ = apply_actions_logic(cells, (4, 4), 5, ENV)  # stay, stay
    assert rw[0] >= 10.0 and info["tasks"][0] == 1

def test_cooperation_reward():
    """Manhattan distance 1 without collision -> both +2."""
    cells = [(2, 2), (2, 4)]
    new, rw, info, _ = apply_actions_logic(cells, (4, 2), 5, ENV)  # stay, left -> (2,3)
    assert info["cooperations"] == 1
    assert rw[0] == 2.0 and rw[1] == 2.0


# --- Metrics (reused from E2) ---

def test_fairness_import():
    from nash_depin.fairness_metrics import gini_coefficient
    assert gini_coefficient([10.0, 10.0]) < 1e-6
    assert gini_coefficient([20.0, 0.0]) > 0.4


# --- Solver with equity (reused from E2) ---

def test_solver_equity():
    from nash_depin.nash_solver import joint_nash
    A = np.array([[0.0, 10.0], [2.0, 0.0]])
    B = np.array([[0.0, 2.0], [10.0, 0.0]])
    s0, s1, V0, V1 = joint_nash(A, B, fast=True, equity=True)
    assert abs(s0.sum() - 1) < 1e-6 and abs(s1.sum() - 1) < 1e-6
