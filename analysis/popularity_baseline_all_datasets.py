# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import os, sys
import numpy as np, pandas as pd
MANUL = _paths.REPO_ROOT
sys.path.insert(0, _paths.NCF_DIR)
from data_preparation_interactions import prepare_sequences, subsample_users_items, train_val_test_split_next_item
from ncf_negative_sampling_datasets import NCFTestDatasetSampled
TOP_K = 10
DATA = _paths.DATA_DIR

def ev(score, ds):
    items = np.asarray(ds.items); labels = np.asarray(ds.labels)
    st = np.flatnonzero(labels == 1.0); en = np.append(st[1:], len(labels))
    hr, nd = [], []
    for a, b in zip(st, en):
        c = score[items[a:b]]
        r = int((c > c[0]).sum())
        hr.append(1.0 if r < TOP_K else 0.0)
        nd.append(1.0 / np.log2(r + 2) if r < TOP_K else 0.0)
    return float(np.mean(hr)), float(np.mean(nd))

DS = [("ML-1M", "ml-1m", None), ("Amazon Beauty", "amazon_beauty", "Beauty_and_Personal_Care"),
      ("Arts", "amazon_arts_crafts_sewing", "Arts_Crafts_and_Sewing"),
      ("Grocery", "amazon_grocery_gourmet_food", "Grocery_and_Gourmet_Food"),
      ("Office", "amazon_office_products", "Office_Products")]

def main():
    for name, d, cat in DS:
        if cat is None:
            df = pd.read_csv(os.path.join(DATA, d, "ratings.dat"), sep="::", engine="python",
                             header=None, names=["userId","movieId","rating","timestamp"])
        else:
            df = pd.read_csv(os.path.join(DATA, d, cat + ".csv")).rename(
                columns={"user_id":"userId","parent_asin":"movieId"})[["userId","movieId","rating","timestamp"]]
        dfm, _ = prepare_sequences(df)
        print(f"\n### {name}")
        for (N, M) in [(300,800),(1000,1200),(3000,2500)]:
            out = []
            for msl, tag in ((2, "NCF"), (5, "SASRec")):
                _, u2s, nu, ni = subsample_users_items(dfm.copy(), max_users=N, max_movies=M, min_seq_len=msl)
                tr, va, te = train_val_test_split_next_item(u2s, min_len=3)
                upt = {u: set() for u in range(nu)}
                for (u, m, _r) in tr: upt[int(u)].add(int(m))
                uht = {u: set(v) for u, v in upt.items()}
                for (u, _l, vi) in va: uht[int(u)].add(int(vi))
                ds = NCFTestDatasetSampled(next_triples=te, num_items=ni, user_pos_all_set=uht, num_ng=99, seed=456)
                pop = np.zeros(ni)
                for (_u, m, _r) in tr: pop[int(m)] += 1
                hr, nd = ev(pop, ds)
                out.append(f"{tag}-protocol {hr:.4f}/{nd:.4f}")
            print(f"  {N}/{M}\t" + "\t".join(out), flush=True)


if __name__ == "__main__":
    main()
