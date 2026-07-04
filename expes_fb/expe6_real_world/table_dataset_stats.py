"""Generate a dataset-statistics table for Experiment 6.

The table covers the union of datasets currently used in Settings 1 and 2 and
reports, for each dataset:

  - which setting(s) use it
  - number of samples
  - number of features
  - density (% non-zero entries)
  - average non-zeros per sample
  - positive-class rate

Usage
-----
    python table_dataset_stats.py
    python expes_fb/expe6_real_world/table_dataset_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import issparse


SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_loaders import get_dataset
import run_s1
import run_s2


RESULTS_DIR = SCRIPT_DIR / "results"
TABLE_PATH = RESULTS_DIR / "table_dataset_stats.tex"
DATA_DIR = SCRIPT_DIR / "data"

DATASET_DISPLAY = {
    "mnist": r"\textsc{mnist} (0/1)",
    "phishing": r"\textsc{phishing}",
    "rcv1": r"\textsc{rcv1}",
    "rcv1.binary": r"\textsc{rcv1.binary}",
    "rcv1_train.binary": r"\textsc{rcv1.binary}",
    "real-sim": r"\textsc{real-sim}",
    "w8a": r"\textsc{w8a}",
    "news20.binary": r"\textsc{news20}",
}


def _dataset_usage(name):
    used_in = []
    if name in run_s1.DATASETS:
        used_in.append("S1")
    if name in run_s2.DATASETS:
        used_in.append("S2")
    return "+".join(used_in)


def _dataset_names():
    names = []
    for name in list(run_s1.DATASETS) + list(run_s2.DATASETS):
        if name not in names:
            names.append(name)
    return names


def _density(X):
    n_samples, n_features = X.shape
    total = int(n_samples) * int(n_features)
    if total == 0:
        return 0.0
    if issparse(X):
        nnz = int(X.nnz)
    else:
        nnz = int(np.count_nonzero(X))
    return nnz / total


def _nnz(X):
    if issparse(X):
        return int(X.nnz)
    return int(np.count_nonzero(X))


def _collect_rows():
    rows = []
    for name in _dataset_names():
        X, y = get_dataset(name, DATA_DIR)
        n_samples, n_features = X.shape
        nnz = _nnz(X)
        rows.append(
            dict(
                dataset=name,
                used_in=_dataset_usage(name),
                n_samples=int(n_samples),
                n_features=int(n_features),
                nnz=nnz,
                density=float(_density(X)),
                avg_nnz_per_sample=float(nnz / max(int(n_samples), 1)),
                pos_rate=float(np.mean(y == 1)),
            )
        )
    return rows


def _fmt_int(value):
    return f"{int(value):,}"


def _fmt_density(value):
    pct = 100.0 * float(value)
    if pct >= 10.0:
        spec = ".1f"
    elif pct >= 1.0:
        spec = ".2f"
    else:
        spec = ".3f"
    return rf"{pct:{spec}}\%"


def _fmt_float(value):
    value = float(value)
    if value >= 100:
        spec = ".0f"
    elif value >= 10:
        spec = ".1f"
    else:
        spec = ".2f"
    return f"{value:{spec}}"


def build_table(rows):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \small")
    lines.append(r"  \caption{%")
    lines.append(r"    Dataset characteristics for Experiment~6.")
    lines.append(r"    Density is the percentage of non-zero entries in the feature matrix.")
    lines.append(r"    Avg.~nnz/sample denotes the average number of non-zero features per example.")
    lines.append(r"  }")
    lines.append(r"  \label{tab:expe6_dataset_stats}")
    lines.append(r"  \begin{tabular}{llrrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Dataset & Used in & Samples & Features & Density (\%) & Avg.~nnz/sample & Pos.~rate (\%) \\"
    )
    lines.append(r"    \midrule")

    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                [
                    DATASET_DISPLAY.get(row["dataset"], row["dataset"]),
                    row["used_in"],
                    _fmt_int(row["n_samples"]),
                    _fmt_int(row["n_features"]),
                    _fmt_density(row["density"]),
                    _fmt_float(row["avg_nnz_per_sample"]),
                    _fmt_density(row["pos_rate"]),
                ]
            )
            + r" \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    rows = _collect_rows()
    table = build_table(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table + "\n")

    print(f"Loaded {len(rows)} datasets.")
    for row in rows:
        print(
            f"{row['dataset']:16s} "
            f"used_in={row['used_in']:<5s} "
            f"n={_fmt_int(row['n_samples']):>10s} "
            f"p={_fmt_int(row['n_features']):>10s} "
            f"density={100.0 * row['density']:.4f}% "
            f"avg_nnz/sample={row['avg_nnz_per_sample']:.2f} "
            f"pos_rate={100.0 * row['pos_rate']:.2f}%"
        )
    print()
    print(f"Saved {TABLE_PATH}")
    print()
    print(table)


if __name__ == "__main__":
    main()
