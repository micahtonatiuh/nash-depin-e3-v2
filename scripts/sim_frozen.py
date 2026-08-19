#!/usr/bin/env python3
"""
sim_frozen.py — Pure simulation of the frozen policy (no ROS).

Replays a transferred E2 Q-table under the SAME decision logic used on the
robots (experiments/run_e5.py), producing the reference trajectory that the
physical runs are compared against by experiments/compare_traj.py.

The policy is deterministic here: epsilon is forced to 0, so act() takes
argmax(sigma) of the egalitarian equilibrium. This is what makes the
sim-vs-physical comparison an exact logical-fidelity test rather than a
distributional one.

Usage (from the repository root):
    python3 scripts/sim_frozen.py --seed 2
    python3 scripts/sim_frozen.py --seed 4 --config config/e5_config_tb4x2.yaml

Output: sim_traj_seed{SEED}.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from nash_depin.nash_solver import joint_nash
from experiments.run_e5 import encode_state, apply_actions_logic, PhysicalNashTeam

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="E2 seed of the transferred Q-table")
    ap.add_argument("--config", default="config/e5_config.yaml")
    ap.add_argument("--out", default=None, help="Default: sim_traj_seed{SEED}.json")
    args = ap.parse_args()

    seed = args.seed
    np.random.seed(seed)

    cfg = yaml.safe_load(open(args.config))
    cfg["qtables"]["agent_0"] = f"qtables_e5/seed_{seed}_agent_0.npz"
    cfg["qtables"]["agent_1"] = f"qtables_e5/seed_{seed}_agent_1.npz"

    env = cfg["environment"]
    grid = cfg["physical"]["grid_size"]

    team = PhysicalNashTeam(cfg["qtables"]["agent_0"], cfg["qtables"]["agent_1"], cfg)
    team.epsilon = 0.0   # frozen: deterministic argmax policy

    cells = [tuple(c) for c in env["start_cells"]]
    state = encode_state(cells, grid)
    initial_state = state

    s0, s1, V0, V1 = joint_nash(team.q[0][state], team.q[1][state],
                                fast=True, equity=True)
    print("sigma0:", s0, "\nsigma1:", s1, "\nV0,V1:", V0, V1)

    traj = []
    for step in range(env["steps_per_episode"]):
        actions = team.act(state)
        new_cells, rewards, info, mover = apply_actions_logic(cells, actions, grid, env)
        log.info(f"step {step}: cells={cells} act={actions} -> "
                 f"{new_cells} move={mover} col={info['collisions']}")
        traj.append({"step": step, "cells": [list(c) for c in cells],
                     "act": list(actions), "new_cells": [list(c) for c in new_cells],
                     "col": info["collisions"]})
        cells = new_cells
        state = encode_state(cells, grid)

    out_path = Path(args.out or f"sim_traj_seed{seed}.json")
    out_path.write_text(json.dumps({
        "seed": seed,
        "initial_state": initial_state,
        "sigma0": s0.tolist() if hasattr(s0, "tolist") else list(s0),
        "sigma1": s1.tolist() if hasattr(s1, "tolist") else list(s1),
        "V0": float(V0), "V1": float(V1),
        "steps": traj,
    }, indent=2))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
