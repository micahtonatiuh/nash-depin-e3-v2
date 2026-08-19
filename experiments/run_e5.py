"""
run_e5.py — Orchestrator for the physical robot runs (no blockchain).
Nash-DePIN.

Runs on the lab PC (rclpy). Loads the E2 Q-tables, computes the joint state,
selects actions (Nash policy with egalitarian refinement), translates them
into physical motion and logs fairness metrics.

Semantics IDENTICAL to the E2 simulation:
  - State: ((r0*5+c0)*25) + (r1*5+c1) — same encoding as environment_mixed
  - Task: fixed station at cell (0,0), immediate respawn on completion
  - Collision: if both target the same cell (or swap cells), NEITHER moves and
    both receive -5. Resolved BEFORE moving, so physical contact is impossible
    by construction.
  - Cooperation: Manhattan distance == 1 without collision -> both +2

Phases:
  frozen   (eps 1-20): learning OFF; eps 1-5 at reduced speed (safety)
  finetune (eps 21-50): learning ON with reduced alpha + equity=True

In the frozen phase epsilon is forced to 0 and act() takes argmax(sigma),
making each rollout deterministic. This is what allows the physical runs to
be compared step-by-step against scripts/sim_frozen.py.

Usage:
  python3 experiments/run_e5.py --config config/e5_config.yaml --output e5_results/
  python3 experiments/run_e5.py ... --resume          # resume after a battery pause
  python3 experiments/run_e5.py ... --dry_run         # 2 eps, validates the pipeline

ROS imports are deferred into main() so that the pure decision logic
(encode_state, apply_actions_logic, PhysicalNashTeam) can be imported without
rclpy installed — used by scripts/sim_frozen.py and tests/.
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from nash_depin.grid_actions import ACTION_DELTA
from nash_depin.nash_solver import joint_nash
from nash_depin.fairness_metrics import gini_coefficient, jain_fairness_index

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Environment logic (exact mirror of environment_mixed.py)            #
# ------------------------------------------------------------------ #

def encode_state(cells, grid):
    """((r0*g+c0)*g^2) + (r1*g+c1) — identical to E2."""
    s = 0
    for f, c in cells:
        s = s * grid * grid + f * grid + c
    return s


def apply_actions_logic(cells, actions, grid, env):
    """
    Replicates environment_mixed.step WITHOUT moving any robot:
    computes target cells, detects collisions (same cell or cell swap),
    assigns rewards and returns (new_cells, rewards, info, move[i]).
    move[i]=False means robot i must NOT be moved physically.
    """
    rewards = [0.0, 0.0]
    info = {"collisions": 0, "tasks": [0, 0], "cooperations": 0}
    targets = []
    for i, a in enumerate(actions):
        df, dc = ACTION_DELTA[a]
        f = max(0, min(grid - 1, cells[i][0] + df))
        c = max(0, min(grid - 1, cells[i][1] + dc))
        targets.append((f, c))

    mover = [targets[i] != cells[i] for i in range(2)]

    # Collision: same target cell, or cell swap
    same_cell = targets[0] == targets[1]
    swap = targets[0] == cells[1] and targets[1] == cells[0]
    if same_cell or swap:
        rewards[0] += env["collision_penalty"]
        rewards[1] += env["collision_penalty"]
        info["collisions"] = 1
        targets = [cells[0], cells[1]]   # both revert
        mover = [False, False]
    
    # Task: fixed station (same as E2 with num_tasks=1)
    task = tuple(env["task_cell"])
    for i in range(2):
        if targets[i] == task:
            rewards[i] += env["task_reward"]
            info["tasks"][i] = 1
            break   # one robot per step, same as the simulation

    # Spatial cooperation
    if info["collisions"] == 0:
        man = abs(targets[0][0] - targets[1][0]) + abs(targets[0][1] - targets[1][1])
        if 0 < man <= env["coop_distance"]:
            rewards[0] += env["cooperation_reward"]
            rewards[1] += env["cooperation_reward"]
            info["cooperations"] = 1

    return targets, rewards, info, mover


# ------------------------------------------------------------------ #
# Physical Nash agent                                                 #
# ------------------------------------------------------------------ #

class PhysicalNashTeam:
    """Two E2 Q-tables + egalitarian Nash policy. Learning optional."""

    def __init__(self, q0_path, q1_path, cfg):
        lp = cfg["learning"]
        self.q = [self._load(q0_path), self._load(q1_path)]
        self.alpha = lp["alpha_finetune"]
        self.gamma = lp["gamma"]
        self.epsilon = lp["epsilon"]
        self.equity = lp["use_equity"]
        self.learning = False
        self.n_actions = self.q[0].shape[1]

    @staticmethod
    def _load(path):
        p = Path(path)
        if p.suffix == ".npz":
            return np.load(p)["q_table"]
        if p.suffix == ".npy":
            return np.load(p)
        import pickle
        with open(p, "rb") as f:
            data = pickle.load(f)
        return data["q_table"] if isinstance(data, dict) else data

    def act(self, state):
        s0, s1, _, _ = joint_nash(self.q[0][state], self.q[1][state],
                                  fast=True, equity=self.equity)
        acts = []
        for i, sigma in enumerate([s0, s1]):
            if np.random.random() < self.epsilon:
                acts.append(np.random.randint(self.n_actions))
            else:
                # frozen (eps=0): deterministic policy = modal action of the mixed equilibrium
                if self.epsilon == 0.0:
                    acts.append(int(np.argmax(sigma)))
                else:
                    acts.append(int(np.random.choice(self.n_actions, p=sigma)))
        return tuple(acts)

    def learn(self, state, actions, rewards, next_state):
        if not self.learning:
            return
        _, _, V0, V1 = joint_nash(self.q[0][next_state], self.q[1][next_state],
                                  fast=True, equity=self.equity)
        nash_next = [V0, V1]
        for i in range(2):
            cur = self.q[i][state, actions[0], actions[1]]
            target = rewards[i] + self.gamma * nash_next[i]
            self.q[i][state, actions[0], actions[1]] += self.alpha * (target - cur)

    def save(self, outdir, episode):
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            np.savez(outdir / f"agent_{i}_ep{episode}.npz", q_table=self.q[i])


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/e5_config.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from state.json (after a battery pause)")
    ap.add_argument("--dry_run", action="store_true", help="2 test episodes")
    ap.add_argument("--phase", choices=["frozen", "finetune", "both"], default="both", help="Phase to run (default: both)")
    ap.add_argument("--seed", type=int, default=None, help="Override the Q-table seed (e.g. 3)")
    ap.add_argument("--stochastic", action="store_true",
                    help="Do not seed the RNG: equilibrium-distribution regime")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        if not args.stochastic:
            np.random.seed(args.seed)
        cfg["qtables"]["agent_0"] = f"qtables_e5/seed_{args.seed}_agent_0.npz"
        cfg["qtables"]["agent_1"] = f"qtables_e5/seed_{args.seed}_agent_1.npz"


    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(out / "experiment.log"),
                                  logging.StreamHandler(sys.stdout)])

    env = cfg["environment"]
    phy = cfg["physical"]
    ph = cfg["phases"]
    grid = phy["grid_size"]
    #total_eps = 2 if args.dry_run else ph["frozen_episodes"] + ph["finetune_episodes"]
    if args.dry_run:
        n_frozen = 2 if args.phase in ("frozen", "both") else 0
        n_finetune = 2 if args.phase in ("finetune", "both") else 0
    else:
        n_frozen = ph["frozen_episodes"] if args.phase in ("frozen", "both") else 0
        n_finetune = ph["finetune_episodes"] if args.phase in ("finetune", "both") else 0
    total_eps =n_frozen + n_finetune

    # --- Resume state ---
    state_file = out / "state.json"
    start_ep = 1
    if args.resume and state_file.exists():
        st = json.loads(state_file.read_text())
        start_ep = st["next_episode"]
        log.info(f"RESUMING from episode {start_ep}")

    # --- Nash team ---
    team = PhysicalNashTeam(cfg["qtables"]["agent_0"], cfg["qtables"]["agent_1"], cfg)
    #if args.resume and start_ep > ph["frozen_episodes"]:
    if args.resume and start_ep > n_frozen:
        # load the last fine-tuning Q-tables saved
        last = sorted((out / "qtables").glob("agent_0_ep*.npz"))
        if last:
            ep_tag = last[-1].stem.split("ep")[-1]
            team.q[0] = np.load(out / "qtables" / f"agent_0_ep{ep_tag}.npz")["q_table"]
            team.q[1] = np.load(out / "qtables" / f"agent_1_ep{ep_tag}.npz")["q_table"]
            log.info(f"Q-tables restored from ep{ep_tag}")

    # --- ROS2 (imported here so the pure logic above stays importable
    #     without rclpy; see the module docstring) ---
    import rclpy
    from nash_depin.grid_motion import RobotMotion
    from nash_depin.locobot_motion import LoCoBotMotion
    rclpy.init()
    platforms = cfg.get("robot_platform", ["tb4", "tb4"])
    start_cells = [tuple(c) for c in env["start_cells"]]
    robots = []
    for i, plat in enumerate(platforms):
        if plat == "locobot":
            robots.append(LoCoBotMotion(start_cells[i], cfg))
        else:
            robots.append(RobotMotion(cfg["ros"]["robots"][
            sum(1 for p in platforms[:i] if p == "tb4")]["namespace"], cfg))
    for i, r in enumerate(robots):
        log.info(f"Waiting for odometry from robot {i}...")
        if not r.wait_odom(timeout=100.0):
            log.critical(f"No odometry from robot {i}. Check bringup. Aborting.")
            return 1
    log.info("Odometry OK for both robots")
    cell = phy["cell_size_m"]
    origins = [(-c * cell, f * cell) for (f, c) in env["start_cells"]]

    # --- Episode CSV ---
    csv_path = out / "episode_metrics.csv"
    new_csv = not csv_path.exists() or not args.resume
    f_csv = open(csv_path, "a" if args.resume else "w", newline="")
    writer = csv.writer(f_csv)
    if new_csv:
        writer.writerow(["episode", "phase", "r0", "r1", "tasks0", "tasks1",
                         "collisions", "cooperations", "cell_err0_m", "cell_err1_m",
                         "duration_s"])

    log.info("=" * 60)
    log.info(f"Physical run | eps {start_ep}..{total_eps}")
    log.info(f"Arena: {grid}x{grid} cells of {phy['cell_size_m']} m | "
             f"fixed task at {env['task_cell']}")
    log.info("=" * 60)

    try:
        for ep in range(start_ep, total_eps + 1):
            frozen = ep <= n_frozen
            team.learning = not frozen
            team.epsilon = 0.0 if frozen else cfg["learning"]["epsilon"]
            phase = "frozen" if frozen else "finetune"
            safety = ep <= ph["safety_episodes"]
            speed = 0.10 if safety else None

            log.info(f"\n--- Episode {ep}/{total_eps} [{phase}]"
                     f"{' (SAFETY SPEED)' if safety else ''} ---")
            log.info("Place the robots on their start cells "
                     f"{env['start_cells']} and press ENTER...")

            input()
            # Refresh odometry after physical repositioning.
            for r in robots:
                for _ in range(10):
                    rclpy.spin_once(r, timeout_sec=0.05)
                # Re-anchor the origin per episode (LoCoBot only): the robot is
                # physically on its start_cell whatever the absolute odom reads.
                for i, r in enumerate(robots):
                    if platforms[i] == "locobot":
                        row0, col0 = env["start_cells"][i]
                        r.origin_x = r.x - col0 * cell
                        r.origin_y = r.y + row0 * cell
                        origins[i] = (r.origin_x, r.origin_y)
                for i, r in enumerate(robots):
                    print(f"[EP-START] robot{i} x={r.x:.3f} y={r.y:.3f} cell={r.current_cell(*origins[i])}")                 
            
            cells = [tuple(c) for c in env["start_cells"]]
            state = encode_state(cells, grid)
            cum_r = [0.0, 0.0]; cum_t = [0, 0]; cum_col = 0; cum_coop = 0; recent = []; prev_t = [0, 0]; stale = 0
            t0 = time.time()
            traj = []

            for step in range(env["steps_per_episode"]):
                actions = team.act(state)
                new_cells, rewards, info, mover = apply_actions_logic(
                    cells, actions, grid, env)
                log.debug(f"paso {step}: cells={cells} act={actions} -> {new_cells} "
                         f"mover={mover} col={info['collisions']}")
                traj.append({"step": step, "cells": [list(c) for c in cells],
                             "act": list(actions), "new_cells": [list(c) for c in new_cells],
                             "col": info["collisions"]})
                             
                # Cycle detector: cum_t staleness (a productive monopoly keeps
                recent.append(state)
                if len(recent) > 30:
                    recent.pop(0)
                if cum_t == prev_t:
                    stale += 1
                else:
                    stale = 0; prev_t = list(cum_t)
                if stale >= 40:
                    log.info(f"Cycle/stall at step {step} "
                             f"(cum_t unchanged for {stale} steps); ending episode.")
                    break            

                # Move only the robots that should move (collision = nobody)
                # Physical order: the robot vacating the cell moves first (avoids in-transit crossing)
                order = [0, 1]
                if new_cells[0] == cells[1] and mover[1]:
                    order = [1, 0]
                elif new_cells[1] == cells[0] and mover[0]:
                    order = [0, 1]
                
                for i in order:
                    if mover[i]:
                        ok = robots[i].execute_grid_action(actions[i], speed=speed)
                        if not ok:
                            log.warning(f"Robot {i} did not complete the primitive "
                                        f"(step {step}). Continuing with the logical cell.")

                next_state = encode_state(new_cells, grid)
                team.learn(state, actions, rewards, next_state)

                cells = new_cells
                state = next_state
                for i in range(2):
                    cum_r[i] += rewards[i]
                    cum_t[i] += info["tasks"][i]
                cum_col += info["collisions"]
                cum_coop += info["cooperations"]

            # Cell error at episode close (drift check)
            errs = [robots[i].cell_error_m(*cells[i], *origins[i]) for i in range(2)]
            dur = time.time() - t0
            writer.writerow([ep, phase, round(cum_r[0], 1), round(cum_r[1], 1),
                             cum_t[0], cum_t[1], cum_col, cum_coop,
                             round(errs[0], 3), round(errs[1], 3), round(dur, 1)])
            f_csv.flush()

            log.info(f"Ep {ep} [{phase}] R=({cum_r[0]:.0f},{cum_r[1]:.0f}) "
                     f"tasks=({cum_t[0]},{cum_t[1]}) col={cum_col} "
                     f"err=({errs[0]:.2f},{errs[1]:.2f}) m | {dur/60:.1f} min")

            (out / f"traj_seed{args.seed}_ep{ep}.json").write_text(json.dumps({
                "seed": args.seed, "phase": phase,
                "cell_error_m": [round(e, 3) for e in errs],
                "tasks": cum_t, "collisions": cum_col,
                "steps": traj}, indent=2))

            if max(errs) > phy["cell_error_max_m"]:
                log.warning(f"Cell error > {phy['cell_error_max_m']} m. "
                            "Reposition the robots manually before the next episode.")

            # Periodic save + resume state
            if team.learning and ep % ph["save_every"] == 0:
                team.save(out / "qtables", ep)
            state_file.write_text(json.dumps({"next_episode": ep + 1}))

    except KeyboardInterrupt:
        log.info("\nPause requested (Ctrl+C). State saved — resume with --resume.")
        team.save(out / "qtables", 999)
    finally:
        f_csv.close()
        for r in robots:
            r.destroy_node()
        rclpy.shutdown()

    # --- Summary ---
    _summarize(out, cfg)
    return 0


def _summarize(out, cfg):
    rows = list(csv.DictReader(open(out / "episode_metrics.csv")))
    if not rows:
        return
    for phase in ["frozen", "finetune"]:
        sub = [r for r in rows if r["phase"] == phase]
        if not sub:
            continue
        ginis = [gini_coefficient([float(r["r0"]), float(r["r1"])]) for r in sub]
        succ = [1.0 if (int(r["tasks0"]) + int(r["tasks1"])) > 0 else 0.0 for r in sub]
        cols = [int(r["collisions"]) for r in sub]
        log.info(f"\nSUMMARY {phase.upper()} ({len(sub)} eps): "
                 f"Gini median={np.median(ginis):.4f} IQR=({np.percentile(ginis,25):.3f},"
                 f"{np.percentile(ginis,75):.3f}) | success={np.mean(succ):.2f} | "
                 f"col/ep={np.mean(cols):.2f}")
        ok = np.median(ginis) < cfg["hypothesis"]["gini_median_max"]
        log.info(f"{'PASS' if ok else 'FAIL'}: median Gini "
                 f"{'<' if ok else '>='} {cfg['hypothesis']['gini_median_max']}")
    summary = {"episodes": len(rows),
               "end_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
