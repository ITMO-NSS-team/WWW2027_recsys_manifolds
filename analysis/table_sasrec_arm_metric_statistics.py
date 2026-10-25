"""Statistics behind the SASRec results section.

Every comparison is a two-sample Welch t-test on three seeds per arm. With n=3
per side that is a weak test, so it is only ever used to say "indistinguishable"
or "distinguishable at this cell", never to rank arms by p-value. The aggregate
claims lean on a sign test over the 15 dataset x scale cells instead, which does
not depend on the within-cell n.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_paths.TABLES_DIR, "sasrec_metrics_table.csv")
DATASETS = ["MovieLens-1M", "Amazon Beauty", "Arts & Crafts", "Grocery", "Office Products"]
SCALES = ["300x800", "1000x1200", "3000x2500"]
ALPHA = 0.05


def vals(df, ds, sc, arm, metric):
    return df[(df.dataset == ds) & (df.scale == sc) & (df.arm == arm)][metric].values


def compare(df, a, b, metric):
    """a vs b over all 15 cells. Returns per-cell rows + the sign-test summary."""
    rows = []
    for ds in DATASETS:
        for sc in SCALES:
            va, vb = vals(df, ds, sc, a, metric), vals(df, ds, sc, b, metric)
            if len(va) < 2 or len(vb) < 2:
                continue
            t, p = stats.ttest_ind(va, vb, equal_var=False)
            rows.append(dict(dataset=ds, scale=sc, mean_a=va.mean(), mean_b=vb.mean(),
                             diff=va.mean() - vb.mean(), p=p,
                             sig=p < ALPHA, wins=va.mean() > vb.mean()))
    r = pd.DataFrame(rows)
    wins = int(r.wins.sum())
    # sign test over cells: is the direction of the difference consistent?
    sign_p = stats.binomtest(wins, len(r), 0.5).pvalue
    return r, wins, len(r), sign_p


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def report(df, a, b, metric, label):
    r, wins, n, sp = compare(df, a, b, metric)
    sig = r[r.sig]
    print(f"\n{label} [{metric}]")
    print(f"  cell wins: {wins}/{n}   sign test p={sp:.3f}")
    print(f"  mean difference: {r['diff'].mean():+.4f}  (median {r['diff'].median():+.4f})")
    print(f"  significant cells (p<{ALPHA}): {len(sig)}/{n}")
    for _, x in sig.iterrows():
        d = "higher" if x["diff"] > 0 else "lower"
        print(f"      {x.dataset:16s} {x.scale:10s} {x['diff']:+.4f} {d}, p={x.p:.4f}")
    return r


def main():
    df = pd.read_csv(CSV)

    section("1. SUMMARY: mean +- std per arm (HR@10 / NDCG@10)")
    for ds in DATASETS:
        print(f"\n{ds}")
        for sc in SCALES:
            parts = []
            for arm in ["classic", "hypsasrec", "ours32", "ours32idf", "poincare"]:
                h, n_ = vals(df, ds, sc, arm, "hr"), vals(df, ds, sc, arm, "ndcg")
                parts.append(f"{arm}={h.mean():.3f}/{n_.mean():.3f}")
            pop = vals(df, ds, sc, "popularity", "hr")
            print(f"  {sc:10s} " + "  ".join(parts) + f"  pop={pop.mean():.3f}")

    section("2. HypSASRec against classic SASRec")
    for m in ("hr", "ndcg"):
        report(df, "hypsasrec", "classic", m, "HypSASRec vs classic")

    section("3. Our method (raw initialisation) against classic")
    for m in ("hr", "ndcg"):
        report(df, "ours32", "classic", m, "ours32 vs classic")

    section("4. Our method with IDF initialisation against classic")
    for m in ("hr", "ndcg"):
        report(df, "ours32idf", "classic", m, "ours32idf vs classic")

    section("5. IDF initialisation against raw (our method against itself)")
    for m in ("hr", "ndcg"):
        report(df, "ours32idf", "ours32", m, "ours32idf vs ours32")

    section("6. Fixed Poincare against most-popular")
    below = tot = 0
    for ds in DATASETS:
        line = []
        for sc in SCALES:
            p = vals(df, ds, sc, "poincare", "hr").mean()
            q = vals(df, ds, sc, "popularity", "hr").mean()
            tot += 1
            below += p < q
            line.append(f"{sc}: {p:.3f} vs {q:.3f} {'LOWER' if p < q else 'higher'}")
        print(f"  {ds:16s} " + " | ".join(line))
    print(f"\n  Poincare below most-popular in {below} of {tot} cells")

    section("7. Best arm per cell (HR@10), and whether its margin is significant")
    tally = {}
    for ds in DATASETS:
        for sc in SCALES:
            arms = ["classic", "hypsasrec", "ours32", "ours32idf"]
            ms = {a: vals(df, ds, sc, a, "hr") for a in arms}
            order = sorted(arms, key=lambda a: -ms[a].mean())
            best, second = order[0], order[1]
            t, p = stats.ttest_ind(ms[best], ms[second], equal_var=False)
            tally[best] = tally.get(best, 0) + 1
            mark = "significant" if p < ALPHA else "within noise"
            print(f"  {ds:16s} {sc:10s} best={best:10s} "
                  f"(+{ms[best].mean() - ms[second].mean():.4f} over {second}), p={p:.3f} {mark}")
    print("\n  cell wins:", ", ".join(f"{k}={v}" for k, v in
                                             sorted(tally.items(), key=lambda x: -x[1])))

    section("8. Does the best geometry depend on the dataset (two-factor view)")
    sub = df[df.arm.isin(["classic", "hypsasrec", "ours32", "ours32idf"])].copy()
    for sc in SCALES:
        s = sub[sub.scale == sc]
        # does the arm ranking change across datasets at a fixed scale?
        piv = s.groupby(["dataset", "arm"]).hr.mean().unstack()
        ranks = piv.rank(axis=1, ascending=False)
        print(f"\n  scale {sc}: arm ranks across datasets")
        print(piv.round(4).to_string().replace("\n", "\n    "))
        print("    rank of the best arm across datasets:",
              ", ".join(f"{d}={piv.loc[d].idxmax()}" for d in piv.index))
        # Kendall W-style check: are the rankings concordant across datasets?
        print("    rank disagreement (std of rank per arm):",
              ", ".join(f"{a}={ranks[a].std():.2f}" for a in ranks.columns))

    section("9. Correlation of HR@10 and NDCG@10 (why only HR is plotted)")
    s = df[df.arm != "popularity"]
    r_p = stats.pearsonr(s.hr, s.ndcg)
    r_s = stats.spearmanr(s.hr, s.ndcg)
    print(f"  Pearson r={r_p[0]:.4f} (p={r_p[1]:.2e}), Spearman rho={r_s[0]:.4f}")
    agree = dis = 0
    for ds in DATASETS:
        for sc in SCALES:
            arms = ["classic", "hypsasrec", "ours32", "ours32idf"]
            bh = max(arms, key=lambda a: vals(df, ds, sc, a, "hr").mean())
            bn = max(arms, key=lambda a: vals(df, ds, sc, a, "ndcg").mean())
            agree += bh == bn
            dis += bh != bn
            if bh != bn:
                print(f"    disagree: {ds} {sc}: HR->{bh}, NDCG->{bn}")
    print(f"  best-by-HR equals best-by-NDCG in {agree} of {agree + dis} cells")


if __name__ == "__main__":
    main()
