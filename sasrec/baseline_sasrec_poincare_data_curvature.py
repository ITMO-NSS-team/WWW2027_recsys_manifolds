"""
Poincare-MDS arm with a DATA-DERIVED curvature and latent_dim=32.

Why a new file rather than a flag on baseline_sasrec_poincare_data_curvature.py: the
curvature is hardcoded there, two levels down. `poincare_baseline_sasrec.
fit_poincare_embeddings` builds `PoincareBall(c=1.0)` and rescales the MDS
targets by a bare `* 3.0`. Both have to change together - with a small c the
ball's coordinate radius is 1/sqrt(c), so leaving the target scale at 3.0
would fit the points deep inside a huge ball and the curvature would do
nothing. The NCF-side `ncf/baseline_ncf_poincare_fixed_geometry.py` already solved
this (targets scaled to 3/sqrt(c)); this file ports that fix to the SASRec
harness without modifying the shared modules.

Two changes against the previous configuration of this arm:

  * latent_dim 64 -> 32, so that the three geometry arms (ours-32,
    HypSASRec, Poincare) all carry item geometry of the same dimension, and
    so that the already-computed rank-32 curvatures apply by construction -
    delta_rel, and therefore c, depend on the SVD rank, and the rank has to
    match the dimension the geometry is fitted in.
  * c = 1.0 -> the value the authors' procedure yields for this
    (dataset, scale): c = (delta_rel(Poincare disk) / mean delta_rel(X))^2
    with the anchor at tol=1e-12. See results/measures/hypsasrec_curvature.json
    and delta_new_datasets.json.

NOTE both changed at once, deliberately: this is one configuration of the
arm, not an ablation. If the arm's numbers move, the move cannot be
attributed to dimension or curvature separately - a latent-32 run at c=1.0
on one scale would be needed for that.

Everything else is identical to baseline_sasrec_poincare_data_curvature.py: same
build_data_sasrec, same train_and_eval_sasrec_on_fixed_Z harness with
freeze_item_projection, maxlen=30, dropout 0.0, epochs 50, patience 30.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import argparse
import json
import os
import sys
import time

import geoopt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ablation_sasrec_geometry_vs_optimization import (  # noqa: E402
    build_data_sasrec, train_and_eval_sasrec_on_fixed_Z,
)


def fit_poincare_embeddings_c(D_target, dim, device, c, epochs=1000, lr=0.01, seed=0):
    """Hyperbolic MDS at curvature c.

    Ported from ncf/baseline_ncf_poincare_fixed_geometry.py. The ball of curvature c
    has coordinate radius 1/sqrt(c), so the target distances are scaled to
    3/sqrt(c) rather than a fixed 3.0: the fit then sits at the same relative
    position inside the ball at any curvature, comfortably away from the
    boundary where gradients vanish.
    """
    torch.manual_seed(seed)
    n = D_target.shape[0]
    manifold = geoopt.PoincareBall(c=c)

    init = torch.randn(n, dim, device=device) * 1e-3
    z = geoopt.ManifoldParameter(init, manifold=manifold)
    optimizer = geoopt.optim.RiemannianAdam([z], lr=lr)

    iu, ju = torch.triu_indices(n, n, offset=1)
    target = D_target[iu, ju].to(device)
    target = target / target.max() * (3.0 / (c ** 0.5))

    print(f"Poincare MDS: {n} points, dim={dim}, c={c:.6f}, "
          f"radius={1.0 / c ** 0.5:.2f}", flush=True)
    for ep in range(epochs):
        optimizer.zero_grad()
        loss = ((manifold.dist(z[iu], z[ju]) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        if ep % 250 == 0:
            print(f"   epoch {ep} loss={loss.item():.6f}", flush=True)

    with torch.no_grad():
        d_final = manifold.dist(z[iu], z[ju]).cpu().numpy()
        z_tangent = manifold.logmap0(z).detach().cpu().numpy()
        norms = z.norm(dim=-1)
        print(f"   final: mean |z|={norms.mean():.4f} max |z|={norms.max():.4f} "
              f"(radius {1.0 / c ** 0.5:.2f})", flush=True)
    return z_tangent, d_final, (iu.cpu().numpy(), ju.cpu().numpy()), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_users", type=int, default=300)
    p.add_argument("--max_movies", type=int, default=800)
    p.add_argument("--dataset_dir_name", default="amazon_beauty")
    p.add_argument("--dataset_type", default="amazon", choices=["amazon", "movielens"])
    p.add_argument("--amazon_category", default="Beauty_and_Personal_Care")
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--geom_c", type=float, required=True,
                   help="Data-derived curvature for this (dataset, scale), rank-32 delta_rel.")
    p.add_argument("--maxlen", type=int, default=30)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--mds_epochs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--out_dir", default=os.path.join(_paths.SASREC_RUNS_ROOT, "poincare_fixed"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sasrec_config = {"hidden_units": 64, "maxlen": args.maxlen, "num_blocks": 2,
                     "num_heads": 1, "dropout_rate": 0.0}

    tmp_logs = os.path.join(args.out_dir, f"tmp_logs_{args.tag}")
    data = build_data_sasrec(args.max_users, args.max_movies, 5, args.dataset_dir_name,
                             device, tmp_logs, args.dataset_type, args.amazon_category,
                             maxlen=args.maxlen)
    print(f"num_users={data['num_users']} num_movies={data['num_movies']}", flush=True)

    t0 = time.time()
    z_tangent_np, d_final, (iu, ju), n = fit_poincare_embeddings_c(
        data["D_input_init"].cpu(), dim=args.latent_dim, device=device,
        c=args.geom_c, epochs=args.mds_epochs, seed=args.seed)
    fit_s = time.time() - t0

    D_dense = np.zeros((n, n), dtype=np.float64)
    D_dense[iu, ju] = d_final
    D_dense[ju, iu] = d_final
    np.savez(os.path.join(args.out_dir, f"geometry_{args.tag}.npz"), D=D_dense)

    t0 = time.time()
    hr, ndcg = train_and_eval_sasrec_on_fixed_Z(
        torch.tensor(z_tangent_np, device=device), data, device, args.latent_dim,
        sasrec_config, lr=1e-3, epochs=args.epochs, patience=args.patience,
        seed=args.seed, verbose=True)
    train_s = time.time() - t0

    print(f"\n=== POINCARE (c={args.geom_c:.6f}, dim={args.latent_dim}) {args.tag} ===")
    print(f"Test HR@10={hr:.4f}  NDCG@10={ndcg:.4f}")
    with open(os.path.join(args.out_dir, f"results_{args.tag}.json"), "w") as f:
        json.dump({"test_hr": float(hr), "test_ndcg": float(ndcg), "geom_c": args.geom_c,
                   "latent_dim": args.latent_dim, "maxlen": args.maxlen, "seed": args.seed,
                   "max_users": args.max_users, "max_movies": args.max_movies,
                   "mds_fit_s": fit_s, "train_s": train_s}, f, indent=2)


if __name__ == "__main__":
    main()
