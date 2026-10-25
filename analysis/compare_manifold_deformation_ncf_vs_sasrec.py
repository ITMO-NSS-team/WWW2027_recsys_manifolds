"""How much does the learned manifold move during training: NCF vs SASRec.

The claim under test is "NCF deforms the manifold ~8x more than SASRec". Two
things make that comparison easy to get wrong, and both are handled here.

1. Only ml-1m is comparable. Every GI-NCF run is on ml-1m, so the Amazon
   categories are excluded from the headline number rather than pooled in.

2. GI-NCF is read from the paper configuration only (`GINCF/patience_300`:
   lr_isomap=0.001, freeze_item_projection=True, 300 outer epochs), which runs
   the same number of outer epochs as the SASRec arm, so the two share an
   epoch axis. Both the total change and the per-epoch rate are reported, and
   they answer different questions.

Movement is measured as |last - first| / |first| on each diagnostic, which
makes measures with different units comparable. Measures that can legitimately
cross zero (ORC mean, which is signed) are additionally reported as an absolute
change, since a relative change against a near-zero baseline explodes.

The H1 sentinel (-1 = "skipped for safety") is filtered out before any
aggregation, per the README.
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

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = _paths.DIAGNOSTICS_DIR

# the four measures the project treats as reliable, plus lambda_max for scale
MEASURES = ["orc_mean", "spectral_lambda2", "kruskal_stress", "spearman_rho",
            "lat_h0_max_persistence", "geo_h0_max_persistence"]
SIGNED = {"orc_mean"}          # can cross zero: relative change is meaningless
SCALES = ["300x800", "1000x1200", "3000x2500"]


def load(pattern, model):
    rows = []
    for f in sorted(glob.glob(os.path.join(DIAG, pattern))):
        name = os.path.basename(f)
        scale = next((s for s in SCALES if s in name), None)
        if scale is None:
            continue
        df = pd.read_csv(f).sort_values("epoch")
        if len(df) < 3:
            continue
        rec = dict(model=model, scale=scale, run=name, n_epochs=len(df),
                   first_epoch=df.epoch.iloc[0], last_epoch=df.epoch.iloc[-1],
                   span=df.epoch.iloc[-1] - df.epoch.iloc[0])
        for m in MEASURES:
            if m not in df.columns:
                continue
            v = df[m].replace(-1, np.nan).dropna()
            if len(v) < 3:
                continue
            a, b = v.iloc[0], v.iloc[-1]
            rec[f"{m}_abs"] = abs(b - a)
            rec[f"{m}_rel"] = abs(b - a) / abs(a) if abs(a) > 1e-12 else np.nan
            # total path length: how much it moved, not just where it ended up
            rec[f"{m}_path"] = np.abs(np.diff(v.values)).sum()
            rec[f"{m}_path_rel"] = (np.abs(np.diff(v.values)).sum() / abs(a)
                                    if abs(a) > 1e-12 else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ncf_p = load("geometry_diagnostics_ncfpaper_*.csv", "NCF(paper)")
    sas = load("geometry_diagnostics_sasrec_sasrec_ml1m_latent32_*.csv", "SASRec")
    both = pd.concat([ncf_p, sas], ignore_index=True)

    print("=" * 84)
    print("COVERAGE (ml-1m only: every NCF run uses it)")
    print("=" * 84)
    for (mo, sc), g in both.groupby(["model", "scale"]):
        print(f"  {mo:7s} {sc:10s} runs {len(g)}, "
              f"diagnostic points {g.n_epochs.min()}-{g.n_epochs.max()}, "
              f"epochs {g.first_epoch.min():.0f}..{g.last_epoch.max():.0f}")

    for suffix, title in (("rel", "NET displacement |last-first|/|first|"),
                          ("path_rel", "TOTAL PATH sum|diff| / |first|")):
        print("\n" + "=" * 84)
        print(f"{title}  (NCF/SASRec ratio in the last column)")
        print("=" * 84)
        print(f"  {'measure':26s} {'scale':11s} {'NCF paper':>11s} "
              f"{'SASRec':>10s} {'NCF/SASRec':>11s}")
        for m in MEASURES:
            col = f"{m}_{suffix}"
            if col not in both.columns:
                continue
            for sc in SCALES:
                sv = both[(both.model == "SASRec") & (both.scale == sc)][col].dropna()
                if not len(sv):
                    continue
                pv = both[(both.model == "NCF(paper)") & (both.scale == sc)][col].dropna()
                print(f"  {m:26s} {sc:11s} "
                      f"{(pv.mean() if len(pv) else np.nan):11.4f} "
                      f"{sv.mean():10.4f} "
                      f"{(pv.mean() / sv.mean() if len(pv) and sv.mean() > 0 else np.nan):10.1f}x")

    print("\n" + "=" * 84)
    print("SIGNED MEASURES: absolute change (relative is unstable near zero)")
    print("=" * 84)
    for m in sorted(SIGNED):
        for sc in SCALES:
            pv = both[(both.model == "NCF(paper)") & (both.scale == sc)][f"{m}_abs"].dropna()
            sv = both[(both.model == "SASRec") & (both.scale == sc)][f"{m}_abs"].dropna()
            if not len(sv):
                continue
            print(f"  {m:14s} {sc:11s} NCF {pv.mean():.5f}  "
                  f"SASRec {sv.mean():.5f}   -> {pv.mean() / sv.mean():.2f}x")

    print("\n" + "=" * 84)
    print("CORRECTION FOR TRAINING LENGTH: path per outer epoch")
    print("=" * 84)
    print("  NCF(paper) and SASRec both run 300 outer epochs, so no")
    print("  normalisation is needed here - shown for completeness.")
    for m in MEASURES:
        col = f"{m}_path_rel"
        if col not in both.columns:
            continue
        n = both[both.model == "NCF(paper)"]
        s = both[both.model == "SASRec"]
        nr = (n[col] / n.span).dropna().mean()
        sr = (s[col] / s.span).dropna().mean()
        if sr and sr > 0:
            print(f"  {m:26s} NCF {nr:.5f}/epoch  SASRec {sr:.5f}/epoch  "
                  f"{nr / sr:6.1f}x")


if __name__ == "__main__":
    main()
