"""Collects the four geometry arms of SASRec into one table for the overview figure.

Seeds 0-2 only: later seeds were run for our own confidence, not for the paper,
and mixing n per cell would make the error bars incomparable.

File naming differs per arm, and Beauty in particular appears under three
conventions ("amazon_beauty", a bare "amazon" in the HypSASRec histories, and no
dataset marker at all in the earliest classic runs), so dataset detection falls
through to Beauty only after every explicitly tagged dataset has been ruled out.
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
import re

import numpy as np

EXTERNAL = os.path.join(_paths.EXTERNAL_RUNS_DIR, "SASRec_external",
         "sasrec_main")
LOC = _paths.SASREC_RUNS_ROOT
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overview_data.json")
SCALES = ["300x800", "1000x1200", "3000x2500"]
DATASETS = ["ML-1M", "Beauty", "Arts", "Grocery", "Office"]
MAX_SEED = 2

acc = {}


def ds_of(stem):
    if "ml1m" in stem or "ml-1m" in stem:
        return "ML-1M"
    for d in ("arts", "grocery", "office"):
        if d in stem:
            return d.capitalize()
    return "Beauty"


def harvest(pattern, arm, key=None, root=LOC, exclude=None):
    n = 0
    for f in glob.glob(os.path.join(root, pattern)):
        stem = os.path.basename(f)[:-5]
        if exclude and exclude in stem:
            continue
        m_sc = re.search(r"(\d+x\d+)", stem)
        m_sd = re.search(r"seed(\d)", stem)
        if not m_sc or not m_sd or m_sc.group(1) not in SCALES:
            continue
        if int(m_sd.group(1)) > MAX_SEED:
            continue
        d = json.load(open(f, encoding="utf-8"))
        if key:
            d = d.get(key, {})
        if d.get("test_hr") is None:
            continue
        acc.setdefault((ds_of(stem), m_sc.group(1), arm), {})[int(m_sd.group(1))] = (
            d["test_hr"], d["test_ndcg"])
        n += 1
    return n


def main():
    counts = {
        "classic (local)": harvest("results_classic_sasrec_*.json", "euclidean"),
        "classic (external)": harvest("results_classic_sasrec_*.json", "euclidean", root=EXTERNAL),
        "ours-32 (local)": harvest("results_sasrec_*latent32_*.json", "ours",
                                      key="gi_sasrec", exclude="hidden32"),
        "ours-32 (external)": harvest("results_sasrec_*lat32*.json", "ours",
                                  key="gi_sasrec", root=EXTERNAL, exclude="smoketest"),
        "HypSASRec-BCE": harvest("hyp_sasrec_fixed/bce/history_*.json", "hyperbolic"),
        "Poincare-32": harvest("poincare_fixed/results_*.json", "poincare", exclude="smoke"),
    }
    for k, v in counts.items():
        print(f"  {k:22s} {v} files", flush=True)

    rows = []
    for ds in DATASETS:
        for sc in SCALES:
            r = {"dataset": ds, "scale": sc}
            for arm in ("euclidean", "hyperbolic", "poincare", "ours"):
                v = acc.get((ds, sc, arm))
                if not v:
                    continue
                hr = np.array([v[s][0] for s in sorted(v)])
                nd = np.array([v[s][1] for s in sorted(v)])
                r[arm] = {"hr": float(hr.mean()), "ndcg": float(nd.mean()),
                          "sd": float(hr.std(ddof=1)) if len(hr) > 1 else 0.0,
                          "n": int(len(hr))}
            rows.append(r)
    json.dump(rows, open(OUT, "w"), indent=1)

    print(f"\n{'dataset':8s} {'scale':10s} {'Euclid':>12s} {'Hyperb':>12s} "
          f"{'Poincare':>12s} {'Ours':>12s}")
    for r in rows:
        def c(k):
            return f"{r[k]['hr']:.4f}/{r[k]['n']}" if k in r else "—"
        print(f"{r['dataset']:8s} {r['scale']:10s} {c('euclidean'):>12s} "
              f"{c('hyperbolic'):>12s} {c('poincare'):>12s} {c('ours'):>12s}")
    print(f"\n[Save] {OUT}")


if __name__ == "__main__":
    main()
