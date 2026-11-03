#!/usr/bin/env python3
"""
check_refs.py — Compares references generated on the lab PC with the cloud
references in revision_r14/expected/ (redundancy rule: every result is also
produced on the lab PC).

Compares the logical fields only (cells, actions, tasks, collisions, length).
sigma values can differ in the last decimal across NumPy versions and are not
compared.

Usage:
  python3 revision_r14/check_refs.py r14_results/seed0/reference revision_r14/expected/seed0
"""
import json
import sys
from pathlib import Path

FIELDS = ("cells", "act", "new_cells", "col", "task")


def load(d, ep):
    return json.loads((Path(d) / f"ref_ep{ep}.json").read_text())


def main(lab_dir, exp_dir):
    eps = sorted(int(p.stem.split("ep")[1]) for p in Path(exp_dir).glob("ref_ep*.json"))
    ok_all = True
    for ep in eps:
        a, b = load(lab_dir, ep), load(exp_dir, ep)
        ok = (a["n_steps"] == b["n_steps"] and a["tasks"] == b["tasks"]
              and all(sa[k] == sb[k] for sa, sb in zip(a["steps"], b["steps"])
                      for k in FIELDS))
        ok_all &= ok
        print(f"ep {ep:2d}: {'PASS' if ok else 'FAIL'}  steps {a['n_steps']}/{b['n_steps']}"
              f"  tasks {a['tasks']}/{b['tasks']}")
    print("ALL PASS" if ok_all else "SOME FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
