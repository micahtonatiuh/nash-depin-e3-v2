"""
analyze_e5.py — Sim-to-real comparison for the physical runs.
Nash-DePIN.

Compares Gini (median + IQR), task success and collisions between the E2
simulation and the physical frozen / fine-tuning phases.

IMPORTANT — scope of the Gini comparison:
The low Gini reported in E2 is an ENSEMBLE property across seeds trained with
epsilon > 0, not intra-episode fairness. With an indivisible task
(num_tasks=1), a single frozen rollout can only end in monopoly or a cycle, so
its per-episode Gini is not the same quantity as the E2 seed-level Gini and
the two are not directly comparable. The paper therefore reports logical
FIDELITY (experiments/compare_traj.py) as the physical result, and uses
scripts/classify_equilibria.py to summarize equilibrium types. This script is
retained for the phase-level descriptive comparison it produces.

Usage:
  python3 scripts/analyze_e5.py --results e5_det_seed2/ \
      --sim <E2 nash summary.json> --output plots/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from nash_depin.fairness_metrics import gini_coefficient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--sim", required=True, help="summary.json from E2 nash")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # --- Physical ---
    rows = list(csv.DictReader(open(Path(args.results) / "episode_metrics.csv")))
    data = {}
    for phase in ["frozen", "finetune"]:
        sub = [r for r in rows if r["phase"] == phase]
        if sub:
            data[phase] = {
                "ginis": [gini_coefficient([float(r["r0"]), float(r["r1"])]) for r in sub],
                "success": np.mean([1.0 if int(r["tasks0"]) + int(r["tasks1"]) > 0 else 0.0
                                    for r in sub]),
                "collisions": np.mean([int(r["collisions"]) for r in sub]),
            }

    # --- Simulation (per-seed Gini from E2) ---
    sim = json.loads(Path(args.sim).read_text())
    sim_ginis = [s["gini"] for s in sim["per_seed"]]

    # --- Table ---
    print("\n" + "=" * 64)
    print("SIM-TO-REAL COMPARISON (Gini: median [IQR])")
    print("=" * 64)
    sim_med = np.median(sim_ginis)
    print(f"  Simulation E2     : {sim_med:.4f} "
          f"[{np.percentile(sim_ginis,25):.3f}, {np.percentile(sim_ginis,75):.3f}]")
    for phase, d in data.items():
        g = d["ginis"]
        med = np.median(g)
        gap = (med - sim_med) / max(sim_med, 1e-9) * 100
        print(f"  Physical {phase:9s} : {med:.4f} "
              f"[{np.percentile(g,25):.3f}, {np.percentile(g,75):.3f}] "
              f"| gap {gap:+.1f}% | success {d['success']:.2f} "
              f"| col/ep {d['collisions']:.2f}")
    print("=" * 64)

    # --- Boxplot ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Sim E2"] + [f"Physical {p}" for p in data]
    box = [sim_ginis] + [d["ginis"] for d in data.values()]
    bp = ax.boxplot(box, labels=labels, patch_artist=True, widths=0.5, showmeans=True)
    for patch, color in zip(bp["boxes"], ["#378ADD", "#1D9E75", "#BA7517"]):
        patch.set_facecolor(color); patch.set_alpha(0.55)
    ax.axhline(0.15, color="#E24B4A", linestyle="--", linewidth=1,
               label="Preregistered: median < 0.15")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Sim-to-real transfer (egalitarian Nash)",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        plt.savefig(Path(args.output) / f"e5_sim_to_real{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close()
    print(f"Figures saved to {args.output}")


if __name__ == "__main__":
    main()
