"""
GI-NCF (GINCF) on Arts & Crafts: 3 scales x 3 seeds, queued with a GPU-memory gate.

Configuration is copied verbatim from the Amazon Beauty paper runs
(ncf/arms/arm_ncf_beauty_*.py): eta=0.001,
frozen item projection, select_by="hr", cf_epochs=200, patience 30/30,
dropout=0.2. Only outer_epochs differs by scale, matching those runs:
300 at 300x800 and 1000x1200, 100 at 3000x2500.

Why a queue rather than launching everything at once: a GINCF run takes
10-12 hours, and two concurrent hyperbolic-MDS fits at 2500 items are
enough to push a 16 GB card to 13.9 GB and starve anything else on it. Here
the three seeds of one scale run together (that combination is known to
fit), scales run one after
another, and before each batch the queue waits until the card actually has
room. Nothing starts on a full card.

Each seed is a separate subprocess, so one crash does not take the batch
with it, and progress survives this script being killed - completed runs
leave their summary json behind and are skipped on restart.
"""
# --- repository path bootstrap -------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..")))
import repo_paths as _paths  # noqa: E402,F401
# -------------------------------------------------------------------------
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOGS = os.path.join(_paths.RUNS_DIR, "arts_gincf_logs")

DATASET = dict(dataset_dir_name="amazon_arts_crafts_sewing",
               dataset_type="amazon",
               amazon_category="Arts_Crafts_and_Sewing")
SCALES = [(300, 800, 300), (1000, 1200, 300), (3000, 2500, 100)]  # users, items, outer_epochs
SEEDS = [0, 1, 2]

FREE_MIB_REQUIRED = 7000     # a 3-seed batch needs real headroom; refuse to start below this
POLL_S = 300                 # re-check the card every 5 minutes while waiting
MAX_WAIT_S = 12 * 3600       # give up waiting after 12h rather than hanging forever


def gpu_free_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[0]
        used, total = (int(x.strip()) for x in out.split(","))
        return total - used
    except Exception as e:
        print(f"[warn] nvidia-smi unavailable ({e}); assuming no free memory", flush=True)
        return -1


def wait_for_room(tag):
    waited = 0
    while waited < MAX_WAIT_S:
        free = gpu_free_mib()
        if free >= FREE_MIB_REQUIRED:
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: {free} MiB free, starting", flush=True)
            return True
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: {free} MiB free < {FREE_MIB_REQUIRED}, waiting",
              flush=True)
        time.sleep(POLL_S)
        waited += POLL_S
    print(f"[{tag}] no free memory after {MAX_WAIT_S // 3600}h - skipping", flush=True)
    return False


def driver_source(users, items, outer, seed, prefix):
    """One-off driver for a single run. Written as a file (not -c) so the
    multiprocessing workers inside analyze_run re-import a real module - on
    Windows they spawn and re-import, and the __main__ guard below is what
    keeps that from turning into a fork bomb."""
    return f'''import sys, os
sys.path.insert(0, r"{HERE}")
import ncf_outer_loop_sweep_driver as sweep

if __name__ == "__main__":
    sweep.main(
        n_run_prefix="{prefix}",
        max_users={users},
        max_movies={items},
        dataset_dir_name="{DATASET['dataset_dir_name']}",
        dataset_type="{DATASET['dataset_type']}",
        amazon_category="{DATASET['amazon_category']}",
        select_by="hr",
        cf_epochs=200,
        final_cf_epochs=200,
        inner_patience=30,
        final_patience=30,
        outer_epochs={outer},
        eta_values=[0.001],
        dropout=0.2,
        freeze_item_projection=True,
        seed={seed},
        diagnostics_workers=4,
        summary_filename="{prefix}_summary.json",
    )
'''


def main():
    os.makedirs(LOGS, exist_ok=True)
    for users, items, outer in SCALES:
        batch, skipped = [], []
        for seed in SEEDS:
            prefix = f"arts_gincf_{users}x{items}_seed{seed}"
            if os.path.exists(os.path.join(_paths.RUNS_DIR, f"{prefix}_summary.json")):
                skipped.append(prefix)
                continue
            batch.append((seed, prefix, outer, users, items))
        if skipped:
            print(f"[skip] already computed: {', '.join(skipped)}", flush=True)
        if not batch:
            continue

        tag = f"{users}x{items}"
        if not wait_for_room(tag):
            continue

        procs = []
        for seed, prefix, outer_, users_, items_ in batch:
            drv = os.path.join(LOGS, f"_driver_{prefix}.py")
            with open(drv, "w", encoding="utf-8") as f:
                f.write(driver_source(users_, items_, outer_, seed, prefix))
            log = open(os.path.join(LOGS, f"{prefix}.log"), "w", encoding="utf-8")
            # Force UTF-8 on the child: with stdout redirected to a file,
            # Windows Python otherwise encodes using the system locale and a
            # non-ASCII character in a log line kills the run.
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            p = subprocess.Popen([PY, drv], stdout=log, stderr=subprocess.STDOUT,
                                 cwd=HERE, env=env)
            procs.append((prefix, p, log))
            print(f"[{time.strftime('%H:%M:%S')}] started {prefix} (pid {p.pid})", flush=True)
            time.sleep(90)   # stagger: let one run allocate before the next starts

        for prefix, p, log in procs:
            rc = p.wait()
            log.close()
            print(f"[{time.strftime('%H:%M:%S')}] {prefix} finished, exit code {rc}", flush=True)

    print("ARTS_GINCF_QUEUE_DONE", flush=True)


if __name__ == "__main__":
    main()
