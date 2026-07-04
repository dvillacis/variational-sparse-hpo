"""Plot Experiment 1 — slide version (panel A only, horizontal layout).

Produces a single wide figure suited for a conference slide:

  fig_experiment1_slide.pdf

The 4 contour panels are arranged in a single row, with the upper-level
slice ($\\Phi$ vs. $x_2$) as a wider 5th panel on the right.

Usage
-----
    python plot_slide.py
    python expes_fb/expe1_illustrative/plot_slide.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expes_fb.shared.plotting import (
    PALETTE,
    REFERENCE_COLORS,
    apply_plot_style,
    get_method_style,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "results.pkl"
FIG_PATH = RESULTS_DIR / "fig_experiment1_slide.pdf"

PHI_COLORS = {
    "true": get_method_style("true")["color"],
    "be_0.1": get_method_style("be_0.1")["color"],
    "be_0.5": get_method_style("be_0.5")["color"],
}
PHI_LABELS = {
    "true": "True / FB",
    "be_0.1": "BE, $\\gamma=0.1$",
    "be_0.5": "BE, $\\gamma=0.5$",
}

apply_plot_style()


def _contour_levels(Z):
    zmin = float(np.min(Z))
    zhi = float(np.percentile(Z, 92))
    if not np.isfinite(zhi) or zhi <= zmin:
        zhi = zmin + 1.0
    return np.linspace(zmin + 0.05 * (zhi - zmin), zmin + 0.80 * (zhi - zmin), 9)


def _plot_panel_a_horizontal(fig, panel_a):
    gs = fig.add_gridspec(
        2,
        8,
        height_ratios=[1.0, 0.85],
        wspace=0.32,
        hspace=0.55,
    )
    axes_contour = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(4)]
    ax_phi = fig.add_subplot(gs[1, 2:6])

    y1 = panel_a["grid"]["y1"]
    y2 = panel_a["grid"]["y2"]
    Y1, Y2 = np.meshgrid(y1, y2)
    minima = panel_a["minima"]
    fbe_gamma = panel_a["config"]["fbe_gamma"]

    items = [
        ("true", "True Cost"),
        ("be_0.1", "BE Smooth, $\\gamma=0.1$"),
        ("be_0.5", "BE Smooth, $\\gamma=0.5$"),
        ("fbe", f"FB, $\\gamma={fbe_gamma}$"),
    ]

    for ax, (key, title) in zip(axes_contour, items):
        Z = panel_a["grid"][key]
        ax.contour(
            Y1,
            Y2,
            Z,
            levels=_contour_levels(Z),
            colors=PALETTE["charcoal"],
            linewidths=0.85,
        )

        true_min = np.asarray(minima["true"], dtype=float)
        this_min = np.asarray(minima[key], dtype=float)
        ax.scatter(
            true_min[0],
            true_min[1],
            marker="*",
            s=85,
            color=PALETTE["ink"],
            zorder=5,
        )
        if key.startswith("be_"):
            ax.scatter(
                this_min[0],
                this_min[1],
                marker="o",
                s=42,
                color=PHI_COLORS[key],
                edgecolors="white",
                linewidths=0.8,
                zorder=6,
            )
        ax.axhline(0.0, color=REFERENCE_COLORS["guide"], linewidth=0.8, linestyle=":")
        ax.set_aspect("equal")
        ax.set_title(
            f"{title}\nmin = ({this_min[0]:.3f}, {this_min[1]:.3f})"
        )
        ax.set_xlim(y1.min(), y1.max())
        ax.set_ylim(y2.min(), y2.max())
        ax.set_xlabel("$y_1$")

    axes_contour[0].set_ylabel("$y_2$")
    for ax in axes_contour[1:]:
        ax.tick_params(axis="y", labelleft=False)

    phi_slice = panel_a["phi_slice"]
    x2_grid = np.asarray(phi_slice["x2_grid"], dtype=float)
    minima_phi = phi_slice["minima"]

    for key in ["true", "be_0.1", "be_0.5"]:
        values = np.asarray(phi_slice["curves"][key], dtype=float)
        ax_phi.plot(
            x2_grid,
            values,
            linewidth=1.8,
            color=PHI_COLORS[key],
            label=PHI_LABELS[key],
        )
        ax_phi.scatter(
            minima_phi[key]["x2"],
            minima_phi[key]["phi"],
            s=26,
            color=PHI_COLORS[key],
            zorder=5,
        )

    ax_phi.axvline(
        phi_slice["config"]["kink_x2"],
        color=REFERENCE_COLORS["guide"],
        linewidth=1.0,
        linestyle="--",
    )
    ax_phi.set_xlabel("$x_2 = \\log(\\alpha_2)$")
    ax_phi.set_ylabel("$\\Phi(x_2)$")
    ax_phi.legend(
        loc="upper left",
        ncol=3,
        handlelength=1.6,
        columnspacing=1.2,
    )
    ax_phi.text(
        0.01,
        0.05,
        f"kink at $exp(x_2)={phi_slice['config']['kink_alpha2']:.1f}$",
        transform=ax_phi.transAxes,
        fontsize=8,
        color=PALETTE["charcoal"],
    )


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results not found at {RESULTS_PATH}. Run run.py first.")

    with RESULTS_PATH.open("rb") as f:
        results = pickle.load(f)

    fig = plt.figure(figsize=(11.0, 5.0))
    _plot_panel_a_horizontal(fig, results["panel_a"])

    fig.savefig(FIG_PATH, bbox_inches="tight")
    print(f"Saved {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
