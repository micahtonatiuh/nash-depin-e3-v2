#!/usr/bin/env python3
"""
run_e5_tape.py — R1.4 physical test with a pre-recorded random tape
(Paper 1 revision, Access-2026-41887). Nash-DePIN E5.

New file: experiments/run_e5.py and nash_depin/* are NOT modified. The step
logic (encode_state, apply_actions_logic, collision resolution, physical move
order, stale detector) is imported from or copied verbatim from run_e5.py.

What it adds
------------
1. Stochastic policy driven by a tape (see tape_core.py): sigma sampling plus
   epsilon exploration, as in E2 evaluation, but reproducible. The simulation
   reference for every episode is computed BEFORE the robots move and saved
   next to the physical log.
2. Per-step physical check: after every step it reads odometry and logs the
   measured cell and the distance to the commanded cell centre.
3. --loop closed: the next decision uses the MEASURED cells instead of the
   commanded ones. Logical fidelity is then a real test (in run_e5 the state
   always advances from commanded cells, so it is 100% by construction).
4. --manual_check: at the end of each episode the operator types the cell
   each robot is really in (external ground truth, not odometry).

Usage (from the repo root, ~/nash-depin-e5):
  # 1) references only, no robots (check episode lengths and hand-overs)
  python3 revision_r14/run_e5_tape.py --config config/e5_config_tb4x2.yaml \
      --seed 0 --starts 0 0 4 4 --episodes 10 --output r14_results/seed0 --sim_only
  # 2) physical run
  python3 revision_r14/run_e5_tape.py --config config/e5_config_tb4x2.yaml \
      --seed 0 --starts 0 0 4 4 --episodes 10 --loop closed --manual_check \
      --output r14_results/seed0
  # resume after a battery swap
  ... same command ... --resume
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import tape_core as tc  # noqa: E402

log = logging.getLogger("r14")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/e5_config_tb4x2.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--qdir", default="qtables_e5",
                    help="folder with seed_{s}_agent_{i}.npz")
    ap.add_argument("--starts", type=int, nargs=4, default=None,
                    metavar=("R0", "C0", "R1", "C1"),
                    help="start cells; default = environment.start_cells")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=None,
                    help="default = environment.steps_per_episode")
    ap.add_argument("--policy", choices=["sigma", "greedy"], default="sigma")
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--stale_cut", type=int, default=40,
                    help="run_e5 stale detector; 0 disables it")
    ap.add_argument("--loop", choices=["open", "closed"], default="closed")
    ap.add_argument("--manual_check", action="store_true")
    ap.add_argument("--safety_episodes", type=int, default=1,
                    help="first N episodes at 0.10 m/s")
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sim_only", action="store_true",
                    help="write tapes and references, do not touch the robots")
    return ap.parse_args()


# ------------------------------------------------------------------ #
# Preparation: tapes, references, manifest                            #
# ------------------------------------------------------------------ #

def prepare(args, cfg, out):
    env = cfg["environment"]
    grid = cfg["physical"]["grid_size"]
    steps = args.steps or env["steps_per_episode"]
    starts = ([[args.starts[0], args.starts[1]], [args.starts[2], args.starts[3]]]
              if args.starts else env["start_cells"])
    qp = [Path(args.qdir) / f"seed_{args.seed}_agent_{i}.npz" for i in range(2)]
    q0, q1 = tc.load_q(qp[0]), tc.load_q(qp[1])
    greedy = args.policy == "greedy"
    stale_cut = args.stale_cut or None

    refdir = out / "reference"
    refdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in range(1, args.episodes + 1):
        tape = tc.make_tape(args.seed, ep, steps)
        np.save(refdir / f"tape_ep{ep}.npy", tape)
        ref = tc.simulate_episode(q0, q1, env, grid, starts, tape,
                                  args.epsilon, greedy, stale_cut)
        tc.save_json({"seed": args.seed, "episode": ep, **ref},
                     refdir / f"ref_ep{ep}.json")
        rows.append((ep, ref))

    code = [REPO / "experiments" / "run_e5.py", REPO / "nash_depin" / "nash_solver.py",
            HERE / "tape_core.py", Path(__file__).resolve()]
    manifest = {
        "tape_version": tc.TAPE_VERSION,
        "seed": args.seed, "starts": starts, "steps": steps,
        "episodes": args.episodes, "policy": args.policy,
        "epsilon": args.epsilon, "stale_cut": stale_cut, "loop": args.loop,
        "config": str(args.config),
        "environment": env,
        "qtables": {p.name: sha256(p) for p in qp},
        "code": {str(p.relative_to(REPO)): sha256(p) for p in code},
        "numpy": np.__version__, "python": sys.version.split()[0],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tc.save_json(manifest, out / "manifest.json")

    log.info("Reference (simulation) per episode:")
    log.info("  ep  steps  tasks(R0,R1)  hand-overs  moves  collisions  Gini")
    for ep, r in rows:
        log.info(f"  {ep:2d}  {r['n_steps']:5d}  {str(r['tasks']):12s}  "
                 f"{r['handovers']:10d}  {sum(r['moves']):5d}  "
                 f"{r['collisions']:10d}  {tc.gini2(r['reward']):.3f}")
    moves = np.mean([sum(r["moves"]) for _, r in rows])
    log.info(f"Mean moves per episode: {moves:.0f} "
             f"(rough duration: {moves * 4.5 / 60 + 1.5:.0f} min per episode)")
    return q0, q1, starts, steps, greedy, stale_cut


# ------------------------------------------------------------------ #
# Physical run                                                        #
# ------------------------------------------------------------------ #

def ask_cell(prompt):
    while True:
        s = input(prompt).strip()
        if s == "":
            return None
        try:
            f, c = (int(v) for v in s.replace(",", " ").split())
            return [f, c]
        except ValueError:
            print("  Format: row col (e.g. '0 1'), or Enter to skip.")


def run_physical(args, cfg, out, q0, q1, starts, steps, greedy, stale_cut):
    import rclpy
    from experiments.run_e5 import encode_state, apply_actions_logic  # noqa: F401
    from nash_depin.grid_motion import RobotMotion
    from nash_depin.locobot_motion import LoCoBotMotion

    env = dict(cfg["environment"])
    phy = cfg["physical"]
    grid = phy["grid_size"]
    cell = phy["cell_size_m"]

    state_file = out / "state.json"
    start_ep = 1
    if args.resume and state_file.exists():
        start_ep = json.loads(state_file.read_text())["next_episode"]
        log.info(f"RESUMING from episode {start_ep}")

    # --- ROS2: same construction as run_e5.py ---
    rclpy.init()
    platforms = cfg.get("robot_platform", ["tb4", "tb4"])
    start_cells = [tuple(c) for c in starts]
    robots = []
    for i, plat in enumerate(platforms):
        if plat == "locobot":
            robots.append(LoCoBotMotion(start_cells[i], cfg))
        else:
            robots.append(RobotMotion(cfg["ros"]["robots"][
                sum(1 for p in platforms[:i] if p == "tb4")]["namespace"], cfg))
    for i, r in enumerate(robots):
        log.info(f"Waiting for odometry of robot {i}...")
        if not r.wait_odom(timeout=100.0):
            log.critical(f"No odometry from robot {i}. Check bringup. Aborting.")
            rclpy.shutdown()
            return 1
    # run_e5 convention: odometry zero at each robot's start cell
    origins = [(-c * cell, f * cell) for (f, c) in start_cells]

    csv_path = out / "episode_metrics.csv"
    new_csv = not (args.resume and csv_path.exists())
    f_csv = open(csv_path, "w" if new_csv else "a", newline="")
    wr = csv.writer(f_csv)
    if new_csv:
        wr.writerow(["episode", "loop", "steps_ref", "steps_phy",
                     "first_divergence", "logical_match", "cell_track_rate",
                     "err_med_m", "err_max_m", "term_err0_m", "term_err1_m",
                     "manual_ok", "tasks0", "tasks1", "handovers", "collisions",
                     "gini", "duration_s"])

    try:
        for ep in range(start_ep, args.episodes + 1):
            ref = json.loads((out / "reference" / f"ref_ep{ep}.json").read_text())
            tape = np.load(out / "reference" / f"tape_ep{ep}.npy")
            safety = ep <= args.safety_episodes
            speed = 0.10 if safety else None
            log.info(f"\n--- Episode {ep}/{args.episodes} [{args.loop} loop]"
                     f"{' (SAFETY SPEED)' if safety else ''} --- reference: "
                     f"{ref['n_steps']} steps, tasks {ref['tasks']}, "
                     f"{ref['handovers']} hand-overs")
            log.info(f"Place the robots at {starts} and press ENTER...")
            input()
            for r in robots:
                for _ in range(10):
                    rclpy.spin_once(r, timeout_sec=0.05)
            for i, r in enumerate(robots):            # same as run_e5
                if platforms[i] == "locobot":
                    row0, col0 = start_cells[i]
                    r.origin_x = r.x - col0 * cell
                    r.origin_y = r.y + row0 * cell
                    origins[i] = (r.origin_x, r.origin_y)
            start_meas = [list(robots[i].current_cell(*origins[i])) for i in range(2)]
            log.info(f"[EP-START] measured cells {start_meas} (expected {starts})")
            if start_meas != [list(c) for c in starts]:
                log.warning("Measured start cells differ from the expected ones. "
                            "Check placement/odometry before continuing "
                            "(Ctrl+C to stop, ENTER to continue).")
                input()

            cells = [tuple(c) for c in starts]
            cum_r = [0.0, 0.0]; cum_t = [0, 0]; prev_t = [0, 0]; stale = 0
            cum_col = 0
            traj = []
            t0 = time.time()
            for step in range(steps):
                state, actions, sig = tc.decide(q0, q1, cells, grid, tape[step],
                                                args.epsilon, greedy)
                new_cells, rewards, info, mover = apply_actions_logic(
                    cells, actions, grid, env)
                rec = {"step": step, "cells": [list(c) for c in cells],
                       "act": list(actions),
                       "new_cells": [list(c) for c in new_cells],
                       "col": info["collisions"], "task": list(info["tasks"])}
                traj.append(rec)

                # stale detector, identical to run_e5
                if cum_t == prev_t:
                    stale += 1
                else:
                    stale = 0; prev_t = list(cum_t)
                if stale_cut is not None and stale >= stale_cut:
                    rec["cut"] = True
                    log.info(f"Stale detector: cut at step {step}.")
                    break

                # physical move: same order rule as run_e5
                order = [0, 1]
                if new_cells[0] == cells[1] and mover[1]:
                    order = [1, 0]
                elif new_cells[1] == cells[0] and mover[0]:
                    order = [0, 1]
                ok = [True, True]
                for i in order:
                    if mover[i]:
                        ok[i] = robots[i].execute_grid_action(actions[i], speed=speed)
                        if not ok[i]:
                            log.warning(f"Robot {i} did not complete the primitive "
                                        f"(step {step}).")
                for r in robots:
                    for _ in range(3):
                        rclpy.spin_once(r, timeout_sec=0.02)
                meas = [list(robots[i].current_cell(*origins[i])) for i in range(2)]
                err = [round(robots[i].cell_error_m(*new_cells[i], *origins[i]), 3)
                       for i in range(2)]
                rec.update({"meas_cells": meas, "err_m": err, "prim_ok": ok,
                            "t": round(time.time() - t0, 2)})

                for i in range(2):
                    cum_r[i] += rewards[i]
                    cum_t[i] += info["tasks"][i]
                cum_col += info["collisions"]

                if args.loop == "closed":
                    nxt = [tuple(m) for m in meas]
                    valid = (all(0 <= v < grid for m in nxt for v in m)
                             and nxt[0] != nxt[1])
                    if not valid:
                        log.error(f"Measured cells {meas} invalid (outside grid or "
                                  "shared). Stopping the episode for safety.")
                        rec["abort"] = "invalid_measured_cells"
                        break
                    if nxt != [tuple(c) for c in new_cells]:
                        log.warning(f"step {step}: measured {meas} != commanded "
                                    f"{rec['new_cells']} -> policy continues "
                                    "from the measured cells")
                    cells = nxt
                else:
                    cells = [tuple(c) for c in new_cells]

            dur = time.time() - t0
            term = [round(robots[i].cell_error_m(*cells[i], *origins[i]), 3)
                    for i in range(2)]
            manual = None
            if args.manual_check:
                log.info(f"Final commanded cells: {[list(c) for c in cells]}")
                m0 = ask_cell("  Real cell of robot 0 (row col, Enter=skip): ")
                m1 = ask_cell("  Real cell of robot 1 (row col, Enter=skip): ")
                if m0 is not None and m1 is not None:
                    manual = int([m0, m1] == [list(c) for c in cells])

            executed = [s for s in traj if "meas_cells" in s]
            track = [s["meas_cells"] == s["new_cells"] for s in executed]
            errs = [e for s in executed for e in s["err_m"]]
            fd = tc.first_divergence(ref["steps"], traj)
            n_cmp = min(len(ref["steps"]), len(traj))
            match = (fd is None and len(ref["steps"]) == len(traj))
            hand = tc.count_handovers(traj)
            g = tc.gini2(cum_r)
            wr.writerow([ep, args.loop, ref["n_steps"], len(traj),
                         "" if fd is None else fd, int(match),
                         round(float(np.mean(track)), 4) if track else "",
                         round(float(np.median(errs)), 3) if errs else "",
                         round(float(np.max(errs)), 3) if errs else "",
                         term[0], term[1], "" if manual is None else manual,
                         cum_t[0], cum_t[1], hand, cum_col,
                         round(g, 4), round(dur, 1)])
            f_csv.flush()
            tc.save_json({"seed": args.seed, "episode": ep, "loop": args.loop,
                          "tape_version": tc.TAPE_VERSION,
                          "first_divergence": fd, "compared_steps": n_cmp,
                          "terminal_err_m": term, "manual_ok": manual,
                          "tasks": cum_t, "collisions": cum_col,
                          "handovers": hand, "steps": traj},
                         out / f"phy_ep{ep}.json")
            log.info(f"Ep {ep}: logical match {'YES' if match else 'NO'}"
                     f"{'' if fd is None else f' (diverges at step {fd})'} | "
                     f"cell tracking {np.mean(track) if track else float('nan'):.3f} | "
                     f"err med {np.median(errs) if errs else float('nan'):.3f} m | "
                     f"tasks {cum_t} | hand-overs {hand} | {dur / 60:.1f} min")
            if max(term) > phy["cell_error_max_m"]:
                log.warning(f"Terminal cell error > {phy['cell_error_max_m']} m. "
                            "Re-place the robots before the next episode.")
            state_file.write_text(json.dumps({"next_episode": ep + 1}))
    except KeyboardInterrupt:
        log.info("\nPaused (Ctrl+C). Resume with --resume.")
    finally:
        f_csv.close()
        for r in robots:
            r.destroy_node()
        rclpy.shutdown()
    return 0


def main():
    args = parse()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "experiment.log"),
                                  logging.StreamHandler(sys.stdout)])
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.resume and (out / "manifest.json").exists():
        man = json.loads((out / "manifest.json").read_text())
        q0, q1 = (tc.load_q(Path(args.qdir) / f"seed_{args.seed}_agent_{i}.npz")
                  for i in range(2))
        starts, steps = man["starts"], man["steps"]
        greedy, stale_cut = man["policy"] == "greedy", man["stale_cut"]
    else:
        q0, q1, starts, steps, greedy, stale_cut = prepare(args, cfg, out)
    if args.sim_only:
        log.info(f"--sim_only: tapes and references written to {out / 'reference'}")
        return 0
    return run_physical(args, cfg, out, q0, q1, starts, steps, greedy, stale_cut)


if __name__ == "__main__":
    sys.exit(main())
