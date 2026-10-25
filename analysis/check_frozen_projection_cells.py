"""Checks whether the three collapsed raw-init cells moved at all.

The before/after table shows `before == after` to three decimals for
GI-SASRec raw-init on Arts 3000x2500, Office 1000x1200 and Office 3000x2500 -
exactly the three cells where test HR@10 collapsed. Endpoints alone cannot
distinguish "never moved" from "moved and came back", so this reports the
range over the whole trajectory as well.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import glob
import os
import re

import numpy as np
import pandas as pd

DIAG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diagnostics")

CELLS = {
    "Arts raw 3000x2500":    "*sasrec_arts_lat32_3000x2500_seed*.csv",
    "Office raw 1000x1200":  "*sasrec_office_lat32_1000x1200_seed*.csv",
    "Office raw 3000x2500":  "*sasrec_office_lat32_3000x2500_seed*.csv",
    "Arts IDF 3000x2500":    "*sasrec_arts_idf_3000x2500_seed*.csv",
    "Office IDF 3000x2500":  "*sasrec_office_idf_3000x2500_seed*.csv",
    "Grocery raw 3000x2500": "*sasrec_grocery_lat32_3000x2500_seed*.csv",
    "ml1m raw 3000x2500":    "*sasrec_ml1m_latent32_3000x2500_seed*.csv",
}
MEASURES = ["kruskal_stress", "spearman_rho", "orc_mean"]


def main():
    print(f'{"cell":24s} {"measure":16s} {"first":>9s} {"last":>9s} '
          f'{"range":>9s} {"range/first":>11s} {"uniq.":>7s}')
    for lab, pat in CELLS.items():
        # dedupe: the same run can carry two walker prefixes
        seen = {}
        for f in sorted(glob.glob(os.path.join(DIAG, "geometry_diagnostics_" + pat))):
            run = re.sub(r"^geometry_diagnostics_(idf|lat32|ncf|extcat|sasrec)_", "",
                         os.path.basename(f)[:-4])
            seen.setdefault(run, f)
        if not seen:
            print(f"{lab:24s} (no files)")
            continue
        for c in MEASURES:
            first, last, rng, uniq, n = [], [], [], [], []
            for f in seen.values():
                v = pd.read_csv(f).sort_values("epoch")[c].replace(-1, np.nan).dropna().values
                if len(v) < 2:
                    continue
                first.append(v[0]); last.append(v[-1])
                rng.append(v.max() - v.min())
                uniq.append(len(set(np.round(v, 8)))); n.append(len(v))
            f0, l0, r0 = np.mean(first), np.mean(last), np.mean(rng)
            print(f"{lab:24s} {c:16s} {f0:9.4f} {l0:9.4f} {r0:9.5f} "
                  f"{r0 / abs(f0):10.1%} {np.mean(uniq):4.0f}/{np.mean(n):.0f}")
        print()


if __name__ == "__main__":
    main()
