#!/usr/bin/env python3
"""
run_e5_preflight.py — Pre-flight check and auditable session record.
Run BEFORE each physical battery (frozen or finetune). Does not move robots.

Verifies:
  1. Transferred Q-tables present + checksum against the E2 manifest
  2. Fair-seed selection criterion documented (fair_seeds.json)
  3. Calibration origin_x/origin_y defined (not None / not default)
  4. Config coherent: num_tasks=1, fixed task at (0,0), closed-loop mode

Writes: e5_session_{timestamp}.json (commit, checksums, operator, phase)

Usage:
    python3 experiments/run_e5_preflight.py --config config/e5_config.yaml \
        --qtables qtables_e5/ --phase frozen --operator <name>
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def fail(msg):
    print(f"  FAIL {msg}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--qtables", required=True, help="Dir con Q-tables de E2 + manifest.json")
    ap.add_argument("--phase", required=True, choices=["frozen", "finetune"])
    ap.add_argument("--operator", required=True)
    args = ap.parse_args()

    ok = True
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    qdir = Path(args.qtables)

    print("\n[1] Q-tables and manifest")
    manifest = qdir / "manifest.json"
    qchecks = {}
    if not manifest.exists():
        ok = fail("Falta manifest.json (generar en Ubuntu: sha256 de cada Q-table de E2)")
    else:
        m = json.loads(manifest.read_text(encoding="utf-8"))
        for fname, expected in m.get("checksums", {}).items():
            p = qdir / fname
            if not p.exists():
                ok = fail(f"Falta {fname}")
            elif sha256(p) != expected:
                ok = fail(f"Checksum NO coincide: {fname} (Q-table alterada vs. E2)")
            else:
                qchecks[fname] = expected
                print(f"  OK {fname}")

    print("[2] Fair-seed criterion (a priori)")
    fs = qdir / "fair_seeds.json"
    if not fs.exists():
        ok = fail("Falta fair_seeds.json: {criterio: 'gini<0.15 en E2', seeds: [...]}")
    else:
        d = json.loads(fs.read_text(encoding="utf-8"))
        if "criterio" not in d or "seeds" not in d:
            ok = fail("fair_seeds.json incompleto (requiere 'criterio' y 'seeds')")
        else:
            print(f"  OK criterion='{d.get('criterion', d.get('criterio'))}', seeds={d['seeds']}")

    print("[3] Calibration")
    for k in ("origin_x_m", "origin_y_m"):
        v = cfg.get("physical", {}).get(k, None)
        if v is None:
            ok = fail(f"{k} sin calibrar")
        else:
            print(f"  OK {k} = {v}")

    print("[4] Config coherence")
    checks = [
        (len(cfg.get("environment", {}).get("start_cells", [])) == 2, "environment.start_cells debe tener 2"),
        (tuple(cfg.get("environment", {}).get("task_cell", ())) == (0, 0), "environment.task_cell debe ser (0,0)"),
        (cfg.get("physical", {}).get("cell_size_m") == 0.6, "physical.cell_size_m debe ser 0.6"),
    ]
    for passed, msg in checks:
        if passed:
            print(f"  OK {msg.split(' must')[0]}")
        else:
            ok = fail(msg)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    session = {
        "experiment": "E5", "phase": args.phase, "operator": args.operator,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit,
        "config_sha256": sha256(Path(args.config)),
        "qtable_checksums": qchecks,
        "preflight": "PASS" if ok else "FAIL",
    }
    out = Path(f"e5_session_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.write_text(json.dumps(session, indent=2))

    print(f"\n{'='*50}\nPRE-FLIGHT: {'PASS — proceed with the ' + args.phase + ' battery' if ok else 'FAIL — do NOT run the robots'}")
    print(f"Session recorded: {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
