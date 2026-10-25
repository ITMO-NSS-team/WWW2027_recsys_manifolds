"""Which model families carry a per-epoch history usable for the saturation figure.

The np-baseline logs store `geometry_baseline.history` with one entry per outer
epoch, including `val_hr` and `geom_delta_init`. This checks whether the GI-NCF
and GI-SASRec trees carry the same thing under some key, and how many outer
epochs each family actually ran - a shared epoch axis only makes sense if the
horizons are comparable.
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

ROOTS = {
    "np ml-1m": os.path.join(_paths.EXTERNAL_RUNS_DIR, "GI_np_baselines_ml-1m"),
    "np amazon": os.path.join(_paths.EXTERNAL_RUNS_DIR, "GI_np_baseline_amazon"),
    "GI-NCF": os.path.join(_paths.EXTERNAL_RUNS_DIR, "GINCF"),
    "GI-SASRec (share)": os.path.join(_paths.EXTERNAL_RUNS_DIR, "SASRec_external"),
    "GI-SASRec (local)": _paths.SASREC_RUNS_ROOT,
    "GI-NCF local": _paths.NCF_RUNS_DIR,
}
HIST_FIELDS = ["val_hr", "val_ndcg", "geom_delta_init", "knn_churn",
               "grad_nonfinite_count", "hr_val", "test_hr"]


def find_hist(obj, path="", depth=0, hits=None):
    """Any dict named *history* or holding per-epoch lists."""
    if hits is None:
        hits = []
    if depth > 4 or not isinstance(obj, dict):
        return hits
    for k, v in obj.items():
        p = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            lists = {kk: len(vv) for kk, vv in v.items() if isinstance(vv, list)}
            if lists and ("history" in k.lower() or "epoch" in lists):
                hits.append((p, lists))
            find_hist(v, p, depth + 1, hits)
    return hits


def probe(label, root):
    print("=" * 78)
    print(label)
    print("=" * 78)
    if not os.path.isdir(root):
        print("  root missing\n")
        return
    logs = []
    for depth in range(1, 7):
        logs = glob.glob(os.path.join(root, *(["*"] * depth), "run_log.json"))
        if logs:
            break
    if not logs:
        others = set()
        for depth in range(1, 5):
            for p in glob.glob(os.path.join(root, *(["*"] * depth), "*.json")):
                others.add(os.path.basename(p))
        print(f"  run_log.json not found; json files alongside: {sorted(others)[:8]}\n")
        return
    print(f"  logs found: {len(logs)} (depth {depth})")
    with open(sorted(logs)[0]) as f:
        d = json.load(f)
    print(f"  example: {os.path.dirname(sorted(logs)[0])}")
    print(f"  top-level keys: {list(d.keys())}")
    hits = find_hist(d)
    if not hits:
        print("  NO PER-EPOCH HISTORY")
    for p, lists in hits:
        n = lists.get("epoch", max(lists.values()))
        useful = [f for f in HIST_FIELDS if f in lists]
        print(f"  {p}: {n} epochs; of the needed fields present: {useful or 'none'}")
        print(f"      all fields: {sorted(lists)}")
    print()


def main():
    for label, root in ROOTS.items():
        try:
            probe(label, root)
        except Exception as e:
            print(f"{label}: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
