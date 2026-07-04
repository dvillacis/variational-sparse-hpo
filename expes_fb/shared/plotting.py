"""Shared plotting style for ``expes_fb`` figures.

The defaults are tuned for paper figures:
  - compact typography that remains legible in a one-column layout,
  - grayscale-first styling with colorblind-safe accent colors,
  - explicit line styles and markers so curves stay distinguishable when
    printed without color.
"""

from __future__ import annotations

from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np

ONE_COLUMN_WIDTH = 3.35
TWO_COLUMN_WIDTH = 6.85

PALETTE = {
    "ink": "#1A1A1A",
    "charcoal": "#4D4D4D",
    "slate": "#737373",
    "fog": "#BDBDBD",
    "blue": "#0072B2",
    "teal": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

GROUP_COLORS = {
    "signal": PALETTE["blue"],
    "corr_noise": PALETTE["orange"],
    "pure_noise": PALETTE["fog"],
}

REFERENCE_COLORS = {
    "grid": "#E0E0E0",
    "guide": "#9E9E9E",
    "error": PALETTE["charcoal"],
}

BAR_ERROR_KW = {
    "linewidth": 0.9,
    "ecolor": REFERENCE_COLORS["error"],
}

METHOD_STYLES = {
    "scalar": {"color": PALETTE["charcoal"], "linestyle": "--", "marker": "s"},
    "weighted": {"color": PALETTE["blue"], "linestyle": "-", "marker": "o"},
    "NBA-null": {"color": PALETTE["slate"], "linestyle": "--", "marker": "s"},
    "NBA-SC": {"color": PALETTE["blue"], "linestyle": "-", "marker": "o"},
    "NTRBA-null": {"color": PALETTE["charcoal"], "linestyle": ":", "marker": "D"},
    "NTRBA-SC": {"color": PALETTE["ink"], "linestyle": "-", "marker": "^"},
    "dense": {"color": PALETTE["fog"], "linestyle": "-", "marker": "s"},
    "null": {"color": PALETTE["slate"], "linestyle": "--", "marker": "o"},
    "sc": {"color": PALETTE["ink"], "linestyle": "-", "marker": "^"},
    "sparseho_scalar": {
        "color": PALETTE["slate"], "linestyle": ":", "marker": "s"},
    "sparseho_wl1": {
        "color": PALETTE["blue"], "linestyle": "--", "marker": "o"},
    "nba_wl1": {
        "color": PALETTE["teal"], "linestyle": "-", "marker": "D"},
    "ntrba_wl1": {
        "color": PALETTE["ink"], "linestyle": "-", "marker": "^"},
    "scalar_cv": {"color": PALETTE["charcoal"], "linestyle": "--", "marker": "s"},
    "true": {"color": PALETTE["ink"], "linestyle": "-", "marker": None},
    "fbe": {"color": PALETTE["blue"], "linestyle": "-", "marker": None},
    "be_0.1": {"color": PALETTE["teal"], "linestyle": "--", "marker": None},
    "be_0.5": {"color": PALETTE["purple"], "linestyle": ":", "marker": None},
}

_DEFAULT_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 8,
    "figure.titlesize": 9,
    "figure.titleweight": "normal",
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.titlepad": 4.0,
    "axes.labelpad": 2.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
    "axes.prop_cycle": (
        cycler(
            color=[
                PALETTE["ink"],
                PALETTE["charcoal"],
                PALETTE["slate"],
                PALETTE["fog"],
            ]
        )
        + cycler(linestyle=["-", "--", "-.", ":"])
    ),
}


def apply_plot_style():
    """Apply the shared experiment plotting style."""
    plt.rcParams.update(_DEFAULT_STYLE)


def figure_size(width="onecol", *, aspect=0.7, scale=1.0):
    """Return a compact figure size in inches.

    Parameters
    ----------
    width : {"onecol", "twocol"}
        Target paper width.
    aspect : float
        Height / width ratio.
    scale : float
        Additional multiplicative scale factor.
    """
    if width not in {"onecol", "twocol"}:
        raise ValueError(f"Unknown width={width!r}. Use 'onecol' or 'twocol'.")
    base_width = ONE_COLUMN_WIDTH if width == "onecol" else TWO_COLUMN_WIDTH
    fig_width = base_width * float(scale)
    return fig_width, fig_width * float(aspect)


def grid_figure_size(
    nrows,
    ncols,
    *,
    width="onecol",
    panel_aspect=0.78,
    scale=1.0,
    extra_height=0.0,
):
    """Return a subplot-grid figure size with strict paper width.

    The total width is fixed by ``width`` and split evenly across columns.
    Height is derived from the per-panel aspect ratio and the number of rows.
    ``extra_height`` adds room for titles, legends, or dense tick labels.
    """
    fig_width, _ = figure_size(width=width, aspect=1.0, scale=scale)
    panel_width = fig_width / max(int(ncols), 1)
    panel_height = panel_width * float(panel_aspect)
    fig_height = max(int(nrows), 1) * panel_height + float(extra_height)
    return fig_width, fig_height


def get_method_style(name):
    """Return a copy of the registered style for ``name``."""
    style = METHOD_STYLES.get(name)
    if style is None:
        return {
            "color": PALETTE["charcoal"],
            "linestyle": "-",
            "marker": "o",
        }
    return style.copy()


def collect_legend_items(axes):
    """Collect unique legend items from one or more axes."""
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    handles_out = []
    labels_out = []
    seen = set()
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if not label or label.startswith("_") or label in seen:
                continue
            seen.add(label)
            handles_out.append(handle)
            labels_out.append(label)
    return handles_out, labels_out


def add_shared_legend(
    fig,
    axes,
    *,
    handles=None,
    labels=None,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=None,
    **kwargs,
):
    """Add one figure-level legend from a set of axes."""
    if handles is None or labels is None:
        handles, labels = collect_legend_items(axes)
    if not labels:
        return None
    if ncol is None:
        ncol = min(len(labels), 4)
    return fig.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        **kwargs,
    )
