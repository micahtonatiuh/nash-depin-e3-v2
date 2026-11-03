#!/usr/bin/env python3
"""
test_tape.py — Regression test without robots.

1. With policy=greedy and epsilon=0 the tape logic must reproduce, step by
   step, the trajectories logged on hardware in E3 (e5_det_seed2/4, ep1):
   same cells and same actions in all 200 steps.
2. The tape and the decision rule are deterministic (same input, same output).

Usage (from the repo root):  python3 revision_r14/test_tape.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tape_core as tc  # noqa: E402

REPO = HERE.parent


def main():
    cfg = yaml.safe_load(open(REPO / "config" / "e5_config_tb4x2.yaml"))
    env, grid = cfg["environment"], cfg["physical"]["grid_size"]
    ok_all = True
    for seed in (2, 4):
        q0 = tc.load_q(REPO / "qtables_e5" / f"seed_{seed}_agent_0.npz")
        q1 = tc.load_q(REPO / "qtables_e5" / f"seed_{seed}_agent_1.npz")
        ref = tc.simulate_episode(q0, q1, env, grid, env["start_cells"],
                                  tc.make_tape(seed, 1, 200), 0.0, True)
        phy = json.loads((REPO / f"e5_det_seed{seed}" / f"traj_seed{seed}_ep1.json")
                         .read_text())["steps"]
        n = len(phy)
        same = sum(phy[k]["cells"] == ref["steps"][k]["cells"]
                   and phy[k]["act"] == ref["steps"][k]["act"] for k in range(n))
        ok = same == n == ref["n_steps"]
        ok_all &= ok
        print(f"E3 seed {seed}: {same}/{n} steps identical -> {'PASS' if ok else 'FAIL'}")

    t1 = tc.make_tape(0, 3, 200)
    t2 = tc.make_tape(0, 3, 200)
    ok = np.array_equal(t1, t2) and not np.array_equal(t1, tc.make_tape(0, 4, 200))
    ok_all &= ok
    print(f"Tape determinism: {'PASS' if ok else 'FAIL'}")
    sig = np.array([0.1, 0.2, 0.3, 0.4, 0.0])
    picks = [tc.tape_action(sig, 0.9, u, 0.05, False) for u in (0.05, 0.15, 0.45, 0.95)]
    ok = picks == [0, 1, 2, 3] and tc.tape_action(sig, 0.01, 0.99, 0.05, False) == 4
    ok_all &= ok
    print(f"Inverse-CDF / exploration rule: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
