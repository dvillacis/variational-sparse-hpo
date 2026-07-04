"""Recreate soft-thresholding figures with the shared ``expes_fb`` style.

Usage
-----
    python expes_fb/st_plot/plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expes_fb.shared.plotting import (
    PALETTE,
    REFERENCE_COLORS,
    apply_plot_style,
    figure_size,
)

RESULTS_DIR = Path(__file__).parent / "results"
FIG_PATH = RESULTS_DIR / "graph_soft_thresholding.pdf"

THRESHOLDS = (0.2, 1.0, 2.0)
SLICE_STYLES = [
    {"color": PALETTE["ink"], "linestyle": "--", "label": r"$w_i = t(0.2, v_i)$"},
    {"color": PALETTE["charcoal"], "linestyle": "-.", "label": r"$w_i = t(1.0, v_i)$"},
    {"color": PALETTE["slate"], "linestyle": ":", "label": r"$w_i = t(2.0, v_i)$"},
]

apply_plot_style()


def soft_threshold(u, v):
    """Return the scalar or array-valued soft-threshold map."""
    u = np.asarray(u)
    v = np.asarray(v)
    return np.sign(v) * np.maximum(np.abs(v) - u, 0.0)


def _legend_handles():
    """Return the shared legend handles for the combined figure."""
    return [
        Line2D([0], [0], color=PALETTE["charcoal"], linestyle="-", linewidth=1.35, label=r"$|v_i| = u_i$"),
        Line2D([0], [0], color=PALETTE["ink"], linestyle="--", linewidth=1.6, label=r"$u_i = 0.2$"),
        Line2D([0], [0], color=PALETTE["charcoal"], linestyle="-.", linewidth=1.6, label=r"$u_i = 1.0$"),
        Line2D([0], [0], color=PALETTE["slate"], linestyle=":", linewidth=1.6, label=r"$u_i = 2.0$"),
    ]


def draw_soft_threshold_2d(ax):
    """Draw one-dimensional slices ``w = t(u, v)`` for fixed thresholds."""
    v = np.linspace(-3.0, 3.0, 1201)

    for thr, style in zip(THRESHOLDS, SLICE_STYLES):
        ax.plot(
            v,
            soft_threshold(thr, v),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=style["label"],
        )

    ax.axhline(0.0, color=REFERENCE_COLORS["guide"], linewidth=0.9, linestyle="--")
    ax.axvline(0.0, color=REFERENCE_COLORS["guide"], linewidth=0.9, linestyle="--")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xticks(np.arange(-3, 4, 1))
    ax.set_yticks(np.arange(-3, 4, 1))
    ax.set_xlabel(r"$v_i$")
    ax.set_ylabel(r"$w_i$")
    ax.grid(color=REFERENCE_COLORS["grid"], linewidth=0.7)
    ax.set_box_aspect(1.0)


def draw_soft_threshold_3d(ax):
    """Draw the surface ``w = t(u, v)`` with faceted grayscale styling."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    u = np.linspace(0.0, 3.0, 121)
    v = np.linspace(-3.0, 3.0, 241)
    U, V = np.meshgrid(u, v, indexing="xy")
    W = soft_threshold(U, V)

    ax.plot_wireframe(
        U,
        V,
        W,
        rstride=6,
        cstride=8,
        color="#C7C7C7",
        linewidth=0.55,
    )

    ridge_u = np.linspace(0.0, 3.0, 300)
    ridge_v_pos = ridge_u
    ridge_v_neg = -ridge_u
    line_offset = 0.05
    ridge_w = np.full_like(ridge_u, line_offset)
    ax.plot(
        ridge_u,
        ridge_v_pos,
        ridge_w,
        color=PALETTE["charcoal"],
        linewidth=1.35,
        linestyle="-",
    )
    ax.plot(
        ridge_u,
        ridge_v_neg,
        ridge_w,
        color=PALETTE["charcoal"],
        linewidth=1.35,
        linestyle="-",
    )

    for thr, style in zip(THRESHOLDS, SLICE_STYLES):
        curve_v = np.linspace(-3.0, 3.0, 500)
        curve_u = np.full_like(curve_v, thr)
        curve_w = soft_threshold(curve_u, curve_v) + line_offset
        ax.plot(
            curve_u,
            curve_v,
            curve_w,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.6,
        )

    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_zlim(-3.0, 3.0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([-2, 0, 2])
    ax.set_zticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xlabel(r"$u_i$", labelpad=3.0)
    ax.set_ylabel(r"$v_i$", labelpad=4.0)
    ax.set_zlabel(r"$w_i = t(u_i, v_i)$", labelpad=4.0)
    ax.view_init(elev=24, azim=-58)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = REFERENCE_COLORS["grid"]
        axis._axinfo["grid"]["linewidth"] = 0.7

    ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.45))
    ax.yaxis.set_pane_color((0.90, 0.90, 0.90, 0.35))
    ax.zaxis.set_pane_color((0.98, 0.98, 0.98, 0.15))


def plot_soft_threshold():
    """Create a two-panel soft-thresholding figure with a shared legend."""
    fig = plt.figure(figsize=figure_size("twocol", aspect=0.42))
    ax_left = fig.add_axes([0.08, 0.20, 0.25, 0.52])
    ax_right = fig.add_axes([0.38, 0.12, 0.50, 0.66], projection="3d")
    ax_legend = fig.add_axes([0.08, 0.83, 0.80, 0.10])
    ax_legend.axis("off")

    draw_soft_threshold_2d(ax_left)
    draw_soft_threshold_3d(ax_right)

    ax_legend.legend(
        handles=_legend_handles(),
        loc="center",
        ncol=4,
        columnspacing=1.3,
        handlelength=2.2,
        frameon=False,
    )
    fig.savefig(FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_soft_threshold()
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
