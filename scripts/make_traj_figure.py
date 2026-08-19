"""
make_traj_figure.py — Sim-to-real fidelity figure.

Overlays the simulated (reference) trajectory and the physical trajectory on
the 5x5 grid, one panel per seed/platform. Visualizes the logical-fidelity
result reported in the paper.

Usage (from the repository root):
    python3 scripts/make_traj_figure.py \
        --panel "TB4x2, seed 2" sim_traj_seed2.json e5_det_seed2/traj_seed2_ep1.json \
        --panel "LoCoBot+TB4, seed 2" sim_traj_seed2.json e6_results/seed2/traj_seed2_ep1.json \
        --out fig_traj.png

Each --panel takes three values: TITLE SIM_JSON PHY_JSON
"""
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GRID = 5
TASK_CELL = (0, 0)
# Colors consistent with the E1/E2 figures (grayscale + accent)
C_SIM = ["#1f77b4", "#d62728"]   # robot 0, robot 1 (simulacion)
C_PHY = ["#1f77b4", "#d62728"]   # mismo color, distinto estilo de linea


def load_cells(path):
    """Returns [[(r,c) per step] for robot0, for robot1] from steps[]."""
    with open(path) as f:
        d = json.load(f)
    steps = d["steps"]
    # Occupied-cell sequence: initial state + new_cells of each step
    seq0 = [tuple(steps[0]["cells"][0])]
    seq1 = [tuple(steps[0]["cells"][1])]
    for s in steps:
        seq0.append(tuple(s["new_cells"][0]))
        seq1.append(tuple(s["new_cells"][1]))
    return [seq0, seq1], d


def fidelity(sim, phy):
    """Fidelity over the common prefix, counting STEPS (not states).

    Matches compare_traj.py: the resulting state of each step is compared,
    excluding the initial state (identical by construction).
    An N-step episode contributes N comparisons.
    """
    n = min(len(sim[0]), len(phy[0])) - 1  # discount the initial state
    match = sum(
        1 for k in range(1, n + 1)
        if sim[0][k] == phy[0][k] and sim[1][k] == phy[1][k]
    )
    return match, n


def draw_grid(ax):
    for i in range(GRID + 1):
        ax.axhline(i, color="0.85", lw=0.6, zorder=0)
        ax.axvline(i, color="0.85", lw=0.6, zorder=0)
    # Task cell
    r, c = TASK_CELL
    ax.add_patch(plt.Rectangle((c, GRID - 1 - r), 1, 1,
                               facecolor="#ffe08a", edgecolor="#c8a000",
                               lw=1.0, zorder=1))
    ax.text(c + 0.5, GRID - 1 - r + 0.5, "T", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#7a6000", zorder=2)
    ax.set_xlim(0, GRID)
    ax.set_ylim(0, GRID)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("0.5")


def cell_to_xy(cell, jitter=0.0):
    """(row, col) of the grid -> (x, y) of the plot, with jitter to separate robots."""
    r, c = cell
    return c + 0.5 + jitter, (GRID - 1 - r) + 0.5 + jitter


def plot_panel(ax, title, sim_path, phy_path):
    sim, _ = load_cells(sim_path)
    phy, _ = load_cells(phy_path)
    match, n = fidelity(sim, phy)
    n_states = n + 1  # n steps => n+1 states to draw

    draw_grid(ax)

    # Simulated trajectory: solid line, clipped to the common prefix
    for i in (0, 1):
        j = -0.10 if i == 0 else 0.10
        xs, ys = zip(*[cell_to_xy(c, j) for c in sim[i][:n_states]])
        ax.plot(xs, ys, "-", color=C_SIM[i], lw=2.2, alpha=0.35,
                solid_capstyle="round", zorder=3)

    # Physical trajectory: dotted, on top
    for i in (0, 1):
        j = -0.10 if i == 0 else 0.10
        xs, ys = zip(*[cell_to_xy(c, j) for c in phy[i][:n_states]])
        ax.plot(xs, ys, "--", color=C_PHY[i], lw=1.6, dashes=(3, 2), zorder=4)
        ax.plot(xs[0], ys[0], "o", color=C_PHY[i], ms=5, mec="white",
                mew=0.8, zorder=5)
        ax.plot(xs[-1], ys[-1], "s", color=C_PHY[i], ms=5, mec="white",
                mew=0.8, zorder=5)

    pct = 100.0 * match / n if n else 0.0
    ax.set_title(f"{title}\nfidelity {match}/{n} = {pct:.0f}%",
                 fontsize=9, pad=6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", nargs=3, action="append", metavar=("TITLE", "SIM", "PHY"),
                   required=True, help="Titulo, JSON simulacion, JSON fisico")
    p.add_argument("--out", default="fig_traj.png")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    n = len(args.panel)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.0))
    if n == 1:
        axes = [axes]

    for ax, (title, sim_path, phy_path) in zip(axes, args.panel):
        plot_panel(ax, title, sim_path, phy_path)

    handles = [
        Line2D([], [], color="0.35", lw=2.2, alpha=0.5, label="Simulation"),
        Line2D([], [], color="0.35", lw=1.6, ls="--", dashes=(3, 2), label="Physical"),
        Line2D([], [], color=C_SIM[0], lw=2.0, label="Robot 0"),
        Line2D([], [], color=C_SIM[1], lw=2.0, label="Robot 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Written: {args.out}")

    for title, sim_path, phy_path in args.panel:
        sim, _ = load_cells(sim_path)
        phy, _ = load_cells(phy_path)
        m, nn = fidelity(sim, phy)
        print(f"  {title}: {m}/{nn} = {100.0*m/nn:.1f}%")


if __name__ == "__main__":
    main()
