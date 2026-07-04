"""Generate a unified LaTeX table for Experiment 3.

The goal is to tell one coherent story with one row per problem size ``m``:

1. What does the self-consistent (SC) oracle buy over the null oracle?
2. Once the oracle is fixed to DA, what does NTRBA buy over NBA?
3. What absolute outcome does the winning method (NTRBA-SC) achieve?

All summaries are paired by seed and averaged over the 5 seeds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "results.pkl"
TABLE_PATH = RESULTS_DIR / "table_ablation.tex"


def _fmt(mu, sd, spec):
    if np.isnan(mu):
        return r"\text{---}"
    return rf"{mu:{spec}}\,\pm\,{sd:{spec}}"


def _shade(cell):
    return rf"\cellcolor{{gray!15}} ${cell}$"


def _as_cell(cell):
    if cell.startswith(r"\cellcolor"):
        return cell
    return f"${cell}$"


def _summary_stats(series):
    vals = series.to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def _favorable_cell(mu, sd, spec, *, better):
    del better
    return _as_cell(_fmt(mu, sd, spec))


def _pivot_for_m(df: pd.DataFrame, m: int) -> pd.DataFrame:
    sub = df[df.m == m].copy()
    return sub.pivot(
        index="seed",
        columns="method",
        values=["best_val_loss", "final_grad_norm", "hidden_recall", "f1"],
    )


def build_table(df: pd.DataFrame) -> str:
    ms = sorted(int(m) for m in df.m.unique())
    summaries = {}
    for m in ms:
        piv = _pivot_for_m(df, m)

        nba_da_loss = piv[("best_val_loss", "NBA-SC")]
        nba_null_loss = piv[("best_val_loss", "NBA-null")]
        ntrba_da_loss = piv[("best_val_loss", "NTRBA-SC")]
        ntrba_null_loss = piv[("best_val_loss", "NTRBA-null")]

        nba_da_recall = piv[("hidden_recall", "NBA-SC")]
        nba_null_recall = piv[("hidden_recall", "NBA-null")]
        ntrba_da_recall = piv[("hidden_recall", "NTRBA-SC")]
        ntrba_null_recall = piv[("hidden_recall", "NTRBA-null")]

        nba_da_grad = piv[("final_grad_norm", "NBA-SC")]
        ntrba_da_grad = piv[("final_grad_norm", "NTRBA-SC")]

        ntrba_da_f1 = piv[("f1", "NTRBA-SC")]

        summaries[m] = {
            "nba_oracle_loss": _summary_stats(nba_da_loss - nba_null_loss),
            "nba_oracle_recall": _summary_stats(nba_da_recall - nba_null_recall),
            "ntrba_oracle_loss": _summary_stats(ntrba_da_loss - ntrba_null_loss),
            "ntrba_oracle_recall": _summary_stats(ntrba_da_recall - ntrba_null_recall),
            "loss_reduction": _summary_stats(
                100.0 * (1.0 - ntrba_da_loss / nba_da_loss)
            ),
            "stationarity_gain": _summary_stats(nba_da_grad / ntrba_da_grad),
            "recall_gain": _summary_stats(ntrba_da_recall - nba_da_recall),
            "ntrba_da_loss": _summary_stats(ntrba_da_loss),
            "ntrba_da_f1": _summary_stats(ntrba_da_f1),
        }

    row_defs = [
        ("\\texttt{SC} gain under \\texttt{NBA}", r"$\Delta$ val. loss", "nba_oracle_loss", ".3f", lambda mu: mu < 0.0, True),
        ("\\texttt{SC} gain under \\texttt{NBA}", r"$\Delta$ recall", "nba_oracle_recall", ".3f", lambda mu: mu > 0.0, True),
        ("\\texttt{SC} gain under \\texttt{NTRBA}", r"$\Delta$ val. loss", "ntrba_oracle_loss", ".3f", lambda mu: mu < 0.0, True),
        ("\\texttt{SC} gain under \\texttt{NTRBA}", r"$\Delta$ recall", "ntrba_oracle_recall", ".3f", lambda mu: mu > 0.0, True),
        ("\\texttt{NTRBA} gain under \\texttt{SC}", r"loss red. (\%)", "loss_reduction", ".1f", lambda mu: mu > 0.0, True),
        ("\\texttt{NTRBA} gain under \\texttt{SC}", r"stationarity gain ($\times$)", "stationarity_gain", ".1f", lambda mu: mu > 1.0, True),
        ("\\texttt{NTRBA} gain under \\texttt{SC}", r"recall gain", "recall_gain", ".3f", lambda mu: mu > 0.0, True),
        ("\\texttt{NTRBA}-\\texttt{SC} outcome", r"best val. loss", "ntrba_da_loss", ".3f", lambda mu: False, False),
        ("\\texttt{NTRBA}-\\texttt{SC} outcome", r"F1", "ntrba_da_f1", ".3f", lambda mu: False, False),
    ]

    group_counts = {}
    for group_label, *_ in row_defs:
        group_counts[group_label] = group_counts.get(group_label, 0) + 1

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \caption{%")
    lines.append(r"    \textbf{Experiment~3 oracle$\times$optimizer ablation} (mean\,\textpm\,std over 20 seeds).")
    lines.append(r"    Hidden features are biactive at initialization, so the null oracle")
    lines.append(r"    starves them of subgradient signal while the sign-consistent")
    lines.append(r"    (\texttt{SC}) oracle restores that signal. The first two blocks show")
    lines.append(r"    the gain from replacing the null oracle with \texttt{SC} under")
    lines.append(r"    \texttt{NBA} and \texttt{NTRBA}; the third block shows the additional")
    lines.append(r"    gain from replacing \texttt{NBA} with \texttt{NTRBA} once the oracle")
    lines.append(r"    is fixed to \texttt{SC}. The final block anchors the comparison with")
    lines.append(r"    the absolute outcome of \texttt{NTRBA}-\texttt{SC}. Better directions")
    lines.append(r"    are: negative $\Delta$ loss, positive $\Delta$ recall, positive")
    lines.append(r"    loss reduction, stationarity gain greater than $1$, and positive")
    lines.append(r"    recall gain.")
    lines.append(r"  }")
    lines.append(r"  \label{tab:expe3_oracle_ablation}")
    lines.append(r"  \begin{tabular}{ll" + "r" * len(ms) + r"}")
    lines.append(r"    \toprule")
    lines.append(
        "    Story block & Metric & "
        + " & ".join(rf"$p={m}$" for m in ms)
        + r" \\"
    )
    lines.append(r"    \midrule")

    current_group = None
    for group_label, metric_label, key, spec, better, shade_favorable in row_defs:
        if current_group != group_label:
            if current_group is not None:
                lines.append(r"    \midrule")
            current_group = group_label
            group_cell = rf"\multirow{{{group_counts[group_label]}}}{{*}}{{{group_label}}}"
        else:
            group_cell = ""

        row_cells = [group_cell, metric_label]
        for m in ms:
            mu, sd = summaries[m][key]
            if shade_favorable:
                row_cells.append(_favorable_cell(mu, sd, spec, better=better))
            else:
                row_cells.append(_as_cell(_fmt(mu, sd, spec)))
        lines.append("    " + " & ".join(row_cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results not found at {RESULTS_PATH}. Run run.py first."
        )

    df = pd.read_pickle(RESULTS_PATH)
    table = build_table(df)

    RESULTS_DIR.mkdir(exist_ok=True)
    TABLE_PATH.write_text(table + "\n")
    print(f"Loaded {len(df)} rows from {RESULTS_PATH}")
    print(f"Saved {TABLE_PATH}")
    print()
    print(table)


if __name__ == "__main__":
    main()
