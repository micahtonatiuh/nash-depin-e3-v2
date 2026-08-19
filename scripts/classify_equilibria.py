#!/usr/bin/env python3
"""
classify_equilibria.py — Classifies each physical episode by equilibrium type.

Reads an episode_metrics.csv produced by run_e5.py and labels each episode as
monopoly (r0 or r1), balanced, or cycle (no task completed). With an
indivisible task (num_tasks=1) a frozen rollout can only end in monopoly or a
cycle, so this is the intended way to summarize the physical runs.

Usage:
    python3 scripts/classify_equilibria.py e5_det_seed2/episode_metrics.csv
"""
import sys
import csv

csv_path = sys.argv[1]
counts = {"mono_r0": 0, "mono_r1": 0, "balanced": 0, "cycle": 0}
ginis = []


def gini2(a, b):
    return 0.0 if a + b == 0 else abs(a - b) / (2 * (a + b))


rows = list(csv.DictReader(open(csv_path)))
if not rows:
    print("CSV has no episodes.")
    sys.exit()

for r in rows:
    t0, t1 = int(r["tasks0"]), int(r["tasks1"])
    ginis.append(gini2(t0, t1))
    total = t0 + t1
    if total == 0:
        counts["cycle"] += 1
    elif min(t0, t1) / max(t0, t1) > 0.5:
        counts["balanced"] += 1
    elif t0 > t1:
        counts["mono_r0"] += 1
    else:
        counts["mono_r1"] += 1

n = len(rows)
ginis.sort()
med = ginis[n // 2]
q1, q3 = ginis[n // 4], ginis[(3 * n) // 4]
print(f"Episodes: {n}")
for k, v in counts.items():
    print(f"  {k:12s}: {v:2d} ({100*v/n:.0f}%)")
print(f"Gini median: {med:.4f}  IQR=({q1:.3f},{q3:.3f})")
