#!/usr/bin/env python3
"""
compare_traj.py — Logical fidelity between the simulated and physical runs.

Compares the `cells` field (the grid cell OCCUPIED at each step), not `act`.
Two runs therefore agree when the robots occupy the same cells in the same
order, which is the quantity the paper reports as fidelity.

Usage:
    python3 experiments/compare_traj.py sim_traj_seed2.json \
        e5_det_seed2/traj_seed2_ep1.json
"""
import sys
import json
from pathlib import Path

sim_path, phy_path = sys.argv[1], sys.argv[2]
sim = json.loads(Path(sim_path).read_text())["steps"]
phy = json.loads(Path(phy_path).read_text())["steps"]

n = min(len(sim), len(phy))
match = 0
for k in range(n):
    if sim[k]["cells"] == phy[k]["cells"]:
        match += 1
    else:
        print(f"DIVERGES at step {k}: sim={sim[k]['cells']} phy={phy[k]['cells']}")

print(f"\nFidelity: {match}/{n} identical steps = {100*match/n:.1f}%")
print(f"Length: sim={len(sim)} phy={len(phy)}")
