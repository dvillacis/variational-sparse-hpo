"""One-seed debug harness for Experiment 6 Setting 2.

This script reuses the actual Setting 2 code path, but runs a single seed
sequentially and stores the executed configuration plus per-method outputs
for later debugging.

Examples
--------
uv run python expes_fb/expe6_real_world/run_test.py
uv run python expes_fb/expe6_real_world/run_test.py --dataset mnist --n-outer 20
uv run python expes_fb/expe6_real_world/run_test.py --dataset leukemia --tag debug-a
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse

import run_s2
from data_loaders import get_dataset

DEFAULT_DATASET = "leukemia"
DEBUG_DIR = Path(run_s2.RESULTS_DIR) / "debug_runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one-seed debug test for Experiment 6 Setting 2."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name supported by expes_fb.expe6_real_world.data_loaders.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random split seed.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional row subsample for faster profiling.",
    )
    parser.add_argument(
        "--n-outer",
        type=int,
        default=run_s2.N_OUTER,
        help="Number of outer iterations for wl1 methods.",
    )
    parser.add_argument(
        "--inner-tol",
        type=float,
        default=run_s2.INNER_TOL,
        help="Inner solve tolerance.",
    )
    parser.add_argument(
        "--inner-max-iter",
        type=int,
        default=run_s2.INNER_MAX_ITER,
        help="Inner iteration cap used by HeldOutLogistic.get_val_grad.",
    )
    parser.add_argument(
        "--cv-n-alphas",
        type=int,
        default=run_s2.CV_N_ALPHAS,
        help="Number of C values for scalar CV.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix added to the saved debug artifact names.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the debug harness without persisting artifacts.",
    )
    return parser.parse_args()


def _save_debug_artifacts(df, config, total_elapsed):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{config['tag']}" if config["tag"] else ""
    stem = f"{config['dataset']}_seed{config['seed']}_{stamp}{tag}"

    payload = {
        "config": config,
        "total_elapsed": float(total_elapsed),
        "rows": df.to_dict(orient="records"),
    }

    pkl_path = DEBUG_DIR / f"{stem}.pkl"
    csv_path = DEBUG_DIR / f"{stem}_summary.csv"
    json_path = DEBUG_DIR / f"{stem}_meta.json"

    pd.to_pickle(payload, pkl_path)
    df.drop(
        columns=[
            "beta_final", "val_objs", "inner_debug_records",
            "inner_debug_summary", "algo_debug",
            "algo_debug_records", "algo_debug_summary",
        ],
        errors="ignore",
    ).to_csv(
        csv_path, index=False
    )
    json_path.write_text(json.dumps({
        "config": config,
        "total_elapsed": float(total_elapsed),
        "methods": df["method"].tolist(),
    }, indent=2) + "\n")

    return pkl_path, csv_path, json_path


def _print_inner_debug(df):
    if "inner_debug_summary" not in df.columns:
        return
    print("\nInner solve summary")
    for _, row in df.iterrows():
        summary = row.get("inner_debug_summary") or {}
        if not summary:
            continue
        slowest = summary.get("slowest_call", {})
        print(
            f"{row['method']}: "
            f"calls={summary.get('n_calls', 0)} "
            f"total={summary.get('total_elapsed', np.nan):.1f}s "
            f"mean={summary.get('mean_elapsed', np.nan):.2f}s "
            f"max={summary.get('max_elapsed', np.nan):.2f}s "
            f"passes_mean={summary.get('mean_passes', np.nan):.1f} "
            f"passes_max={summary.get('max_passes', np.nan)} "
            f"stop={summary.get('stop_reasons', {})} "
            f"active_set={summary.get('used_dense_active_set', False)} "
            f"active_mean={summary.get('mean_active_size', np.nan):.1f} "
            f"active_max={summary.get('max_active_size', 0)}"
        )
        if slowest:
            print(
                f"  slowest call #{slowest.get('call_index')} "
                f"context={slowest.get('context')} "
                f"elapsed={slowest.get('elapsed', np.nan):.2f}s "
                f"passes={slowest.get('n_passes')} "
                f"support={slowest.get('support_size')} "
                f"stop={slowest.get('stop_reason')} "
                f"full={slowest.get('full_passes', 0)} "
                f"restricted={slowest.get('restricted_passes', 0)} "
                f"active_mean={slowest.get('mean_active_size', np.nan):.1f}"
            )
        algo_debug = row.get("algo_debug") or {}
        algo_debug_summary = row.get("algo_debug_summary") or {}
        if algo_debug_summary:
            print(
                f"  biactive_mean={algo_debug_summary.get('mean_biactive_size', np.nan):.1f} "
                f"biactive_max={algo_debug_summary.get('max_biactive_size', 0)} "
                f"selected_biactive_mean={algo_debug_summary.get('mean_selected_biactive_size', np.nan):.1f} "
                f"selected_support_mean={algo_debug_summary.get('mean_selected_support_size', np.nan):.1f}"
            )
        if algo_debug:
            print(f"  algo_debug={algo_debug}")


def main():
    args = parse_args()

    run_s2.N_SEEDS = 1
    run_s2.N_OUTER = args.n_outer
    run_s2.INNER_TOL = args.inner_tol
    run_s2.INNER_MAX_ITER = args.inner_max_iter
    run_s2.CV_N_ALPHAS = args.cv_n_alphas

    data_dir = Path(run_s2.DATA_DIR)
    data_dir.mkdir(exist_ok=True)

    run_s2._log(
        f"loading {args.dataset} with seed={args.seed} n_outer={run_s2.N_OUTER} "
        f"inner_tol={run_s2.INNER_TOL:.1e} inner_max_iter={run_s2.INNER_MAX_ITER} "
        f"cv_n_alphas={run_s2.CV_N_ALPHAS}"
    )
    X, y = get_dataset(args.dataset, data_dir)

    if args.max_samples is not None and X.shape[0] > args.max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], args.max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        run_s2._log(f"subsampled to n={X.shape[0]}")

    if issparse(X):
        X = X.tocsc()
        density = X.nnz / (X.shape[0] * X.shape[1])
        run_s2._log(
            f"matrix format=csc n={X.shape[0]:,} m={X.shape[1]:,} density={density:.3e}"
        )
    else:
        run_s2._log(f"matrix format=dense n={X.shape[0]:,} m={X.shape[1]:,}")

    t0 = time.time()
    rows = run_s2.run_one(args.dataset, X, y, args.seed, keep_debug=True)
    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    print("\nPer-method timing summary")
    print(
        df[
            [
                "method", "elapsed", "t_per_iter", "n_iter",
                "best_val", "test_f1", "sparsity", "termination",
            ]
        ].to_string(index=False)
    )
    _print_inner_debug(df)
    print(f"\nTotal wall clock: {elapsed:.1f}s")

    if args.no_save:
        return

    config = {
        "dataset": args.dataset,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "n_outer": run_s2.N_OUTER,
        "inner_tol": run_s2.INNER_TOL,
        "inner_max_iter": run_s2.INNER_MAX_ITER,
        "cv_n_alphas": run_s2.CV_N_ALPHAS,
        "tag": args.tag,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    pkl_path, csv_path, json_path = _save_debug_artifacts(df, config, elapsed)
    print("\nSaved debug artifacts:")
    print(pkl_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
