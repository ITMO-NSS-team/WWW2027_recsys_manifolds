"""What actually distinguishes runs that share a combo folder, scale and seed.

The Amazon item_knn tree has six runs per (combo, scale) where there should be
three: 800-802 and 810-812 both carry seeds 0,1,2 at 300x800. The folder name
encodes only knn_k and temp, so some other hyperparameter differs and is
invisible in the path. Selecting "the winning combo, seed 0" therefore picks
one of two different configurations arbitrarily - which is what the earlier
Amazon diagnostics did.

This prints every config field that varies inside a (head, combo, scale, seed)
group, so the cell key can be extended to whatever it is.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INVENTORY = os.path.join(_paths.TABLES_DIR, "np_amazon_inventory.csv")
IGNORE = {"n_run", "seed", "max_users", "max_movies"}


def main():
    inv = pd.read_csv(INVENTORY)
    inv = inv[inv["status"] == "finished"]

    for (head, combo, scale, seed), g in inv.groupby(
            ["head", "combo", "scale", "seed"]):
        if len(g) < 2:
            continue
        cfgs = []
        for rd in g["run_dir"]:
            with open(os.path.join(rd, "run_log.json")) as f:
                cfgs.append(json.load(f)["config"])
        diff = {k for k in cfgs[0]
                if k not in IGNORE and len({str(c.get(k)) for c in cfgs}) > 1}
        print(f"{head} / {combo} / {scale} / seed{seed}: {len(g)} runs")
        print(f"   n_run: {sorted(g['n_run'])}")
        if diff:
            for k in sorted(diff):
                print(f"   {k}: {[c.get(k) for c in cfgs]}")
        else:
            print("   NO CONFIG DIFFERENCES - these are duplicates of one run")
        print(f"   hr: {[round(float(x), 4) for x in g['hr']]}")
        print()


if __name__ == "__main__":
    main()
