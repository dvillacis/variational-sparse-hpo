"""Plot Experiment 1.

Produces a single combined figure:

  fig_experiment1.pdf

Usage
-----
    python plot.py
    python expes_fb/expe1_illustrative/plot.py
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
    figure_size,
    get_method_style,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "results.pkl"
FIG_PATH = RESULTS_DIR / "fig_experiment1.pdf"

METHOD_COLORS = {
    "null": get_method_style("null")["color"],
    "sc": get_method_style("sc")["color"],
}
METHOD_LABELS = {
    "null": "Null oracle",
    "sc": "Self-consistent oracle",
}
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


def _plot_panel_a(fig, spec, panel_a):
    sub = spec.subgridspec(
        3, 2, height_ratios=[1.0, 1.0, 0.85], hspace=0.42, wspace=0.22
    )
    axes = np.array(
        [
            [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])],
            [fig.add_subplot(sub[1, 0]), fig.add_subplot(sub[1, 1])],
        ]
    )
    ax_phi = fig.add_subplot(sub[2, :])
    axes[0, 0].text(
        -0.35,
        1.15,
        "(a)",
        transform=axes[0, 0].transAxes,
        fontsize=9,
        fontweight="normal",
    )

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

    for ax, (key, title) in zip(axes.ravel(), items):
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

    for ax in axes[:, 0]:
        ax.set_ylabel("$y_2$")
    for ax in axes[0, :]:
        ax.tick_params(axis="x", labelbottom=False)
    for ax in axes[1, :]:
        ax.set_xlabel("$y_1$")

    # axes[0, 0].text(
    #     -0.18,
    #     1.15,
    #     "Panel A  FBE preserves the minimizer; BE smoothings do not",
    #     transform=axes[0, 0].transAxes,
    #     fontsize=11,
    #     fontweight="bold",
    # )

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
    # ax_phi.set_title(
    #     "Upper-level slice on coordinate 2\n"
    #     "$\\Phi(x_2)=|y_2^\\star(x_2)| + "
    #     f"{phi_slice['config']['alpha_weight']:.1f}e^{{x_2}}$"
    # )
    ax_phi.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.1),
        ncol=1,
        handlelength=1.6,
        columnspacing=0.8,
    )
    ax_phi.text(
        0.01,
        0.05,
        f"kink at $exp(x_2)={phi_slice['config']['kink_alpha2']:.1f}$",
        transform=ax_phi.transAxes,
        fontsize=8,
        color=PALETTE["charcoal"],
    )


def _plot_panel_b(fig, spec, panel_b):
    sub = spec.subgridspec(3, 1, hspace=0.38)
    axes = [fig.add_subplot(sub[i, 0]) for i in range(3)]
    axes[0].text(
        -0.35,
        1.15,
        "(b)",
        transform=axes[0].transAxes,
        fontsize=9,
        fontweight="normal",
    )

    for method in ["null", "sc"]:
        rows = panel_b["methods"][method]["rows"]
        it = np.array([row["iteration"] for row in rows], dtype=int)
        x2 = np.array([row["x2"] for row in rows], dtype=float)
        beta2 = np.array([row["beta2"] for row in rows], dtype=float)
        loss = np.array([row["val_loss"] for row in rows], dtype=float)

        for ax, values in zip(axes, [x2, beta2, loss]):
            ax.plot(
                it,
                values,
                marker="o",
                markersize=3.5,
                linewidth=1.7,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )

    axes[0].axhline(
        np.log(panel_b["config"]["alpha0"][1]),
        color=REFERENCE_COLORS["grid"],
        linestyle=":",
    )
    axes[0].set_ylabel("$x_2^k$")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        columnspacing=0.8,
        handlelength=1.6,
    )

    axes[1].axhline(0.0, color=REFERENCE_COLORS["grid"], linestyle=":")
    axes[1].axhline(panel_b["config"]["d_val"][1], color=REFERENCE_COLORS["guide"], linestyle="--")
    axes[1].set_ylabel(r"$y_2^\star(x^k)$")

    axes[2].set_ylabel(r"$\Phi(x^k)$")
    axes[2].set_xlabel("Outer iteration")

    for ax in axes:
        ax.set_xlim(0, panel_b["config"]["n_outer"])


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results not found at {RESULTS_PATH}. Run run.py first.")

    with RESULTS_PATH.open("rb") as f:
        results = pickle.load(f)

    fig = plt.figure(figsize=figure_size("twocol", aspect=0.82))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.28)

    _plot_panel_a(fig, outer[0], results["panel_a"])
    _plot_panel_b(fig, outer[1], results["panel_b"])

    # fig.suptitle(
    #     "Experiment 1 — illustrative sanity checks for FBE and biactive selection",
    #     fontsize=12,
    #     y=0.995,
    # )
    fig.savefig(FIG_PATH, bbox_inches="tight")
    print(f"Saved {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
