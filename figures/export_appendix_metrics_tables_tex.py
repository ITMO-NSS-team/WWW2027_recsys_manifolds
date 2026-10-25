"""Builds the appendix metric tables in LaTeX, mean +- std over three seeds.

The spreadsheet lays the results out as one block per seed. Collapsing the
three seeds into mean +- std makes the table about a third of the height, and
the std is the more useful number anyway: on several cells it is larger than
the gap between the arms, which is a thing the appendix should let a reader
check.

One table per metric rather than `HR / NDCG` in a single cell. With mean+std
a combined cell carries four numbers, and six model columns of those do not
fit a two-column page even under \\resizebox.

Arm names follow the main figure exactly - a reader who sees "Learned Manifold
IDF" in the plot must find that same name in the appendix, not "ours 32 idf".

The curvature column is the value actually used by both hyperbolic arms at
that dataset and scale, derived from delta-hyperbolicity at rank 32; see
`measures/`. It is reported once per row rather than repeated inside two
cells, as the spreadsheet does.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_paths.TABLES_DIR, "sasrec_metrics_table.csv")
OUT = os.path.join(_paths.TABLES_DIR, "appendix_metrics_tables.tex")
PAPER = _paths.FIGURES_OUT_DIR

DATASETS = ["MovieLens-1M", "Amazon Beauty", "Arts & Crafts", "Grocery",
            "Office Products"]
SCALES = ["300x800", "1000x1200", "3000x2500"]
SCALE_TEX = {"300x800": "300/800", "1000x1200": "1000/1200",
             "3000x2500": "3000/2500"}
# arm key -> column header, in table order
ARMS = [("popularity", "Popularity"),
        ("classic", "SASRec classic"),
        ("poincare", "Fixed Poincar\\'e"),
        ("ours32", "Learned Manifold"),
        ("ours32idf", "Learned Manifold IDF"),
        ("hypsasrec", "HypSASRec")]
# competitive arms only: popularity is a reference, not a candidate for "best"
COMPETITIVE = {"classic", "poincare", "ours32", "ours32idf", "hypsasrec"}
CURVATURE = {
    "MovieLens-1M":    {"300x800": 0.00533, "1000x1200": 0.00627, "3000x2500": 0.00735},
    "Amazon Beauty":   {"300x800": 0.01073, "1000x1200": 0.01204, "3000x2500": 0.01449},
    "Arts & Crafts":   {"300x800": 0.01075, "1000x1200": 0.01245, "3000x2500": 0.01895},
    "Grocery":         {"300x800": 0.00869, "1000x1200": 0.01159, "3000x2500": 0.02224},
    "Office Products": {"300x800": 0.00963, "1000x1200": 0.01755, "3000x2500": 0.03978},
}
DS_TEX = {d: d.replace("&", "\\&") for d in DATASETS}


def bold_set(df, ds, sc, metric):
    """Arms whose mean +- std interval overlaps the leader's.

    Bolding the single highest mean overstates the result: in nine of fifteen
    HR@10 rows the leader's own std already exceeds its gap to the runner-up,
    so the bold would mark a difference the three seeds do not resolve. Bolding
    the whole overlapping set says the honest thing instead - these arms are
    not separated by this evidence.

    Since the leader has the largest mean, overlap reduces to
    `mean + std >= best_mean - best_std`.

    That rule alone is not monotone in the mean, and the failure is visible:
    on Amazon Beauty 300/800 it bolds Learned Manifold IDF (0.366 +- 0.028) but
    not Learned Manifold (0.369 +- 0.009), whose mean is higher - the wider
    interval reaches the leader and the tighter one does not. Marking a weaker
    arm and skipping a stronger one reads as a typo, so the set is closed
    downward afterwards: anything at least as good as a marked arm is marked
    too.
    """
    stats = {}
    for k in COMPETITIVE:
        v = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == k)][metric].values
        if len(v) > 1:
            stats[k] = (v.mean(), v.std(ddof=1))
    if not stats:
        return set()
    bm, bs = stats[max(stats, key=lambda k: stats[k][0])]
    overlap = {k for k, (m, s) in stats.items() if m + s >= bm - bs}
    floor = min(stats[k][0] for k in overlap)
    return {k for k, (m, _s) in stats.items() if m >= floor}


def cell(df, ds, sc, arm, metric, bold):
    v = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == arm)][metric].values
    if not len(v):
        return "---"
    if len(v) == 1:                      # popularity is deterministic: no spread
        return f"{v[0]:.3f}"
    m, s = v.mean(), v.std(ddof=1)
    if arm in bold:
        return f"\\textbf{{{m:.3f}}}\\,$\\pm$\\,\\textbf{{{s:.3f}}}"
    return f"{m:.3f}\\,$\\pm$\\,{s:.3f}"


def build(df, metric, label, caption):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{llr" + "r" * len(ARMS) + "}",
        "\\toprule",
        "Dataset & Size & $c$ & " + " & ".join(h for _k, h in ARMS) + " \\\\",
        "\\midrule",
    ]
    for di, ds in enumerate(DATASETS):
        for si, sc in enumerate(SCALES):
            bold = bold_set(df, ds, sc, metric)
            head = f"\\multirow{{3}}{{*}}{{{DS_TEX[ds]}}}" if si == 0 else ""
            cells = [cell(df, ds, sc, k, metric, bold) for k, _h in ARMS]
            lines.append(f"{head} & {SCALE_TEX[sc]} & "
                         f"{CURVATURE[ds][sc]:.4f} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if di < len(DATASETS) - 1 else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def main():
    df = pd.read_csv(CSV)
    note = ("Mean $\\pm$ standard deviation over three seeds. "
            "The popularity baseline is deterministic given the split and has "
            "no spread. $c$ is the curvature derived from "
            "$\\delta$-hyperbolicity at rank 32 and is shared by both "
            "hyperbolic arms. \\textbf{Bold} marks every arm whose "
            "mean\\,$\\pm$\\,std interval overlaps that of the best arm in the "
            "row, i.e.\\ the set this evidence does not separate; several rows "
            "have no single winner. The popularity column is a reference, not "
            "a candidate.")
    tex = ("% Generated by figures/appendix_table_tex.py - do not hand-edit.\n"
           "% Requires \\usepackage{booktabs} and \\usepackage{multirow}.\n\n"
           + build(df, "hr", "tab:appendix-hr",
                   "HR@10 for all geometries, datasets and scales. " + note)
           + "\n"
           + build(df, "ndcg", "tab:appendix-ndcg",
                   "NDCG@10 for all geometries, datasets and scales. " + note))

    for path in (OUT, os.path.join(_paths.TABLES_DIR, "appendix_metrics_tables.tex")):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(tex)
            print(f"[Save] {os.path.normpath(path)}")
        except OSError as e:
            print(f"[not written] {path}: {e}")

    # a sanity read-out: where the spread swallows the gap between the top two
    print("\nwhere std exceeds the leader's margin over the runner-up (HR@10):")
    n_sw = 0
    for ds in DATASETS:
        for sc in SCALES:
            ms = {}
            for k in COMPETITIVE:
                v = df[(df.dataset == ds) & (df.scale == sc) & (df.arm == k)].hr
                if len(v):
                    ms[k] = (v.mean(), v.std(ddof=1))
            order = sorted(ms, key=lambda k: -ms[k][0])
            gap = ms[order[0]][0] - ms[order[1]][0]
            sd = ms[order[0]][1]
            if sd > gap:
                n_sw += 1
                print(f"  {ds:16s} {sc:10s} margin {gap:.4f} < std {sd:.4f} "
                      f"({order[0]})")
    print(f"  {n_sw} of 15 cells in total")


if __name__ == "__main__":
    main()
