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

def ev(score, ds):
    items = np.asarray(ds.items); labels = np.asarray(ds.labels)
    st = np.flatnonzero(labels == 1.0); en = np.append(st[1:], len(labels))
    h = []
    for a, b in zip(st, en):
        c = score[items[a:b]]
        h.append(1.0 if int((c > c[0]).sum()) < TOP_K else 0.0)
    return float(np.mean(h))

def main():
    for name, path, cat in [("Arts", "amazon_arts_crafts_sewing", "Arts_Crafts_and_Sewing"),
                            ("Beauty", "amazon_beauty", "Beauty_and_Personal_Care")]:
        df = pd.read_csv(os.path.join(_paths.DATA_DIR, path, cat + ".csv"))
        df = df.rename(columns={"user_id": "userId", "parent_asin": "movieId"})[["userId","movieId","rating","timestamp"]]
        dfm, _ = prepare_sequences(df)
        for (N, M) in [(300, 800), (3000, 2500)]:
            _, u2s, nu, ni = subsample_users_items(dfm.copy(), max_users=N, max_movies=M, min_seq_len=2)
            tr, va, te = train_val_test_split_next_item(u2s, min_len=3)
            upt = {u: set() for u in range(nu)}
            for (u, m, _r) in tr: upt[int(u)].add(int(m))
            uht = {u: set(v) for u, v in upt.items()}
            for (u, _l, vi) in va: uht[int(u)].add(int(vi))
            ds = NCFTestDatasetSampled(next_triples=te, num_items=ni, user_pos_all_set=uht, num_ng=99, seed=456)
            pop_tr = np.zeros(ni)
            for (_u, m, _r) in tr: pop_tr[int(m)] += 1
            pop_full = pop_tr.copy()
            for (_u, _l, t) in te: pop_full[int(t)] += 1
            for (_u, _l, v) in va: pop_full[int(v)] += 1
            print(f"{name} {N}x{M}:  anti-popularity on TRAIN = {ev(-pop_tr, ds):.4f}   "
                  f"anti-popularity on FULL data = {ev(-pop_full, ds):.4f}   "
                  f"(targets with zero train count: {np.mean([pop_tr[int(t)]==0 for (_u,_l,t) in te])*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
