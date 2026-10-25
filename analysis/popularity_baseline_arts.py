"""
Most-popular baseline on Arts & Crafts, under the project's exact evaluation
protocol.

Diagnostic purpose: the NCF Euclidean arm scores 0.055-0.063
HR@10 at 300x800, which is BELOW the 0.100 a random ranking gets under
1-positive-vs-99-sampled-negatives. Below-chance is not what overfitting
looks like - overfitting lands you at chance. The hypothesis is a
popularity/rarity interaction: the leave-one-out target is each user's LAST
interaction, often a long-tail item, while the 99 negatives are sampled
uniformly and therefore include popular ones. Any model that leans on
popularity then ranks the rare positive below many negatives and scores
systematically worse than chance.

If a pure popularity ranker also lands below chance here, the mechanism is
confirmed and the number is a property of the protocol on sparse data, not
a broken baseline - which is reportable, and much better than quietly
comparing against an arm that never learned.

Uses the project's own subsampling, split and candidate sampler (seed 456
for test, exclusion over train+val), so the candidate sets are identical to
every other arm's. No training, no GPU.
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

MANUL = _paths.REPO_ROOT
sys.path.insert(0, _paths.NCF_DIR)

from data_preparation_interactions import (  # noqa: E402
    prepare_sequences, subsample_users_items, train_val_test_split_next_item,
)
from ncf_negative_sampling_datasets import NCFTestDatasetSampled  # noqa: E402

DATA = os.path.join(_paths.DATA_DIR,
                    "amazon_arts_crafts_sewing", "Arts_Crafts_and_Sewing.csv")
SCALES = [(300, 800), (1000, 1200), (3000, 2500)]
MIN_SEQ_LEN = 2   # the NCF pipeline's value (run_eta_outer_sweep passes min_seq_len=2)
TOP_K = 10


def evaluate(scorer, dataset, name):
    """scorer: per-item score array.

    The dataset is a FLAT list of candidate rows: each user contributes one
    positive (label 1.0) immediately followed by their negatives, and a user
    can have fewer than num_ng negatives when their positive set is dense.
    So blocks are delimited by the positives, not by a fixed stride of 100.
    """
    items = np.asarray(dataset.items)
    labels = np.asarray(dataset.labels)
    starts = np.flatnonzero(labels == 1.0)
    ends = np.append(starts[1:], len(labels))

    hits, ndcgs = [], []
    for s0, s1 in zip(starts, ends):
        cand = items[s0:s1]
        sc = scorer[cand]
        rank = int((sc > sc[0]).sum())   # position 0 of each block is the positive
        if rank < TOP_K:
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
    return float(np.mean(hits)), float(np.mean(ndcgs)), len(hits)


def main():
    print(f"Random ranking gives HR@10 = {TOP_K / 100:.3f}, "
          f"NDCG@10 = {sum(1 / np.log2(r + 1) for r in range(1, TOP_K + 1)) / 100:.4f}\n",
          flush=True)
    df = pd.read_csv(DATA).rename(columns={"user_id": "userId", "parent_asin": "movieId"})
    df = df[["userId", "movieId", "rating", "timestamp"]]
    df_mapped, _ = prepare_sequences(df)

    for (N, M) in SCALES:
        _, u2s, num_users, num_items = subsample_users_items(
            df_mapped.copy(), max_users=N, max_movies=M, min_seq_len=MIN_SEQ_LEN)
        train_events, val_next, test_next = train_val_test_split_next_item(u2s, min_len=3)

        # exclusion set for test candidates: train history + the validation item,
        # matching ablation_geometry_vs_optimization.build_data
        user_pos_train = {u: set() for u in range(num_users)}
        for (u, m, _r) in train_events:
            user_pos_train[int(u)].add(int(m))
        user_hist_test = {u: set(v) for u, v in user_pos_train.items()}
        for (u, _lt, val_item) in val_next:
            user_hist_test[int(u)].add(int(val_item))

        test_ds = NCFTestDatasetSampled(next_triples=test_next, num_items=num_items,
                                        user_pos_all_set=user_hist_test, num_ng=99, seed=456)

        pop = np.zeros(num_items, dtype=np.float64)
        for (_u, m, _r) in train_events:
            pop[int(m)] += 1.0

        hr_pop, nd_pop, n = evaluate(pop, test_ds, "popularity")
        rng = np.random.default_rng(0)
        hr_rnd, nd_rnd, _ = evaluate(rng.random(num_items), test_ds, "random")
        hr_inv, nd_inv, _ = evaluate(-pop, test_ds, "anti-popularity")

        # how rare is the held-out target compared to a uniform negative?
        tgt_pop = np.array([pop[int(t)] for (_u, _l, t) in test_next])
        print(f"=== {N}x{M}: {num_users} users, {num_items} items, "
              f"{len(train_events)} interactions, {n} test users ===", flush=True)
        print(f"    target item popularity: median {np.median(tgt_pop):.0f}, "
              f"mean {tgt_pop.mean():.1f}   |   over the catalogue: median "
              f"{np.median(pop):.0f}, mean {pop.mean():.1f}", flush=True)
        print(f"    most-popular    HR@10={hr_pop:.4f}  NDCG@10={nd_pop:.4f}", flush=True)
        print(f"    random          HR@10={hr_rnd:.4f}  NDCG@10={nd_rnd:.4f}", flush=True)
        print(f"    anti-popular    HR@10={hr_inv:.4f}  NDCG@10={nd_inv:.4f}\n", flush=True)


if __name__ == "__main__":
    main()
