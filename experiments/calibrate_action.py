"""
calibrate_action.py — Isolated test of a single motion primitive.

Validates execute_grid_action(a) on a physical robot (LoCoBot or TB4) and
measures the resulting cell error against the logical cell center. Not part of
the run orchestrator: this is the calibration tool used as a gate before the
cross-hardware runs.

Replicates EXACTLY the robot construction of run_e5.py.

Usage (from the repository root, robot on its start mark):
  python3 experiments/calibrate_action.py --action 0
  python3 experiments/calibrate_action.py --action 0 --speed 0.10 --repeats 3
  python3 experiments/calibrate_action.py --robot 0   # index into start_cells

Gate criterion: error_m < cell_error_max_m (typically 0.10 m).
"""

import argparse
import sys
import time
from pathlib import Path

import yaml
import rclpy

sys.path.insert(0, str(Path(__file__).parent.parent))
from nash_depin.grid_motion import RobotMotion, ACTION_DELTA
from nash_depin.locobot_motion import LoCoBotMotion


def build_robot(idx, cfg):
    """Builds robot idx exactly as run_e5.py does."""
    platforms = cfg.get("robot_platform", ["tb4", "tb4"])
    start_cells = [tuple(c) for c in cfg["environment"]["start_cells"]]
    plat = platforms[idx]
    if plat == "locobot":
        return LoCoBotMotion(start_cells[idx], cfg), start_cells[idx], plat
    ns = cfg["ros"]["robots"][
        sum(1 for p in platforms[:idx] if p == "tb4")]["namespace"]
    return RobotMotion(ns, cfg), start_cells[idx], plat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/e5_config.yaml")
    ap.add_argument("--robot", type=int, default=0,
                    help="Robot index into start_cells (default 0)")
    ap.add_argument("--action", type=int, default=0,
                    help="Action to test (ACTION_DELTA index)")
    ap.add_argument("--speed", type=float, default=0.10,
                    help="Velocidad de seguridad m/s (default 0.10)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Repeticiones de la primitiva (default 1)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cell = cfg["physical"]["cell_size_m"]
    df, dc = ACTION_DELTA[args.action]
    print(f"ACTION_DELTA[{args.action}] = (df={df}, dc={dc})")

    rclpy.init()
    robot, start_cell, plat = build_robot(args.robot, cfg)
    f0, c0 = start_cell
    print(f"Robot {args.robot} | platform={plat} | start cell=({f0},{c0})")

    print("Waiting for odometry...")
    if not robot.wait_odom(timeout=60.0):
        print("ERROR: no odometry. Check bringup / keepalive. Aborting.")
        rclpy.shutdown()
        return 1
    print("Odometry OK")

    # Per-robot origin (identical to run_e5.py: origins = (-c*cell, f*cell))
    origin = (-c0 * cell, f0 * cell)

    # Initial error: the robot should be on its mark -> ~0
    err0 = robot.cell_error_m(f0, c0, *origin)
    print(f"Initial error at cell ({f0},{c0}): {err0:.3f} m "
          f"({'OK' if err0 < cfg['physical']['cell_error_max_m'] else 'FUERA DE MARCA'})")

    # Execute the primitive N times, measuring the error after each one
    f, c = f0, c0
    max_err = err0
    for k in range(args.repeats):
        f, c = f + df, c + dc
        print(f"\n--- Repetition {k+1}/{args.repeats}: execute_grid_action({args.action}) "
              f"-> logical cell ({f},{c}) ---")
        t0 = time.time()
        ok = robot.execute_grid_action(args.action, speed=args.speed)
        dt = time.time() - t0
        err = robot.cell_error_m(f, c, *origin)
        max_err = max(max_err, err)
        flag = "OK" if err < cfg["physical"]["cell_error_max_m"] else "OUT OF TOLERANCE"
        print(f"completed={ok} | duration={dt:.1f}s | error={err:.3f} m  [{flag}]")

    print("\n" + "=" * 50)
    thr = cfg["physical"]["cell_error_max_m"]
    print(f"MAX ERROR: {max_err:.3f} m | threshold: {thr} m")
    print(f"GATE: {'PASS' if max_err < thr else 'FAIL'}")
    print("=" * 50)

    rclpy.shutdown()
    return 0 if max_err < thr else 1


if __name__ == "__main__":
    sys.exit(main())

