"""Checks whether the np-baseline logs carry a per-epoch metric history.

Plotting "quality saturated at epoch X" needs a value per outer epoch, not the
single end-of-run summary. This probes one Amazon run and one ml-1m run and
prints the full key structure of each log, plus anything in the run directory
that could hold a history (csv/json/npy siblings of the snapshots).
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import glob
import json
import os


def walk(obj, prefix="", depth=0, out=None):
    if out is None:
        out = []
    if depth > 3:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                n = len(v)
                kind = "dict" if isinstance(v, dict) else "list"
                out.append(f"{p}  <{kind}, {n}>")
                walk(v, p, depth + 1, out)
            else:
                out.append(f"{p} = {v!r}"[:110])
    elif isinstance(obj, list) and obj:
        out.append(f"{prefix}[0] example: {str(obj[0])[:100]}")
    return out


def probe(label, run_dir):
    print("=" * 78)
    print(f"{label}\n{run_dir}")
    print("=" * 78)
    if not os.path.isdir(run_dir):
        print("  directory missing\n")
        return
    names = sorted(os.listdir(run_dir))
    snaps = [n for n in names if n.startswith("matrices_epoch")]
    other = [n for n in names if not n.startswith("matrices_epoch")]
    print(f"  snapshots: {len(snaps)}")
    print(f"  other files/directories: {other}")
    for sub in other:
        p = os.path.join(run_dir, sub)
        if os.path.isdir(p):
            print(f"    {sub}/ -> {sorted(os.listdir(p))[:8]}")

    log = os.path.join(run_dir, "run_log.json")
    if os.path.exists(log):
        with open(log) as f:
            d = json.load(f)
        print("\n  structure of run_log.json:")
        for line in walk(d):
            print("    " + line)
    print()


def main():
    amz = os.path.join(_paths.EXTERNAL_RUNS_DIR, "GI_np_baseline_amazon")
    ml = os.path.join(_paths.EXTERNAL_RUNS_DIR, "GI_np_baselines_ml-1m")

    one = glob.glob(os.path.join(amz, "*", "*", "*", "run_log.json"))
    if one:
        probe("AMAZON (example run)", os.path.dirname(sorted(one)[0]))

    two = glob.glob(os.path.join(ml, "*", "*", "*", "*", "run_log.json"))
    if two:
        probe("ML-1M (example run, the one with a best/ directory)",
              os.path.dirname(sorted(two)[0]))


if __name__ == "__main__":
    main()
