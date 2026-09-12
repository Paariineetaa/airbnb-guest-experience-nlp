"""
Shared plotting style.

One palette, one set of mark rules, used by every figure in this repo so the
charts read as a single system. Palette is colour-vision-deficiency safe:
hues are assigned in a fixed order and never cycled.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- surfaces & ink -------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e7e6e2"

# --- categorical: fixed order, never cycled -------------------------------
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CAT

# --- sequential: one hue, light -> dark -----------------------------------
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
             "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
             "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)

# --- diverging: two poles + neutral gray midpoint -------------------------
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948", "#8c1f1e"],
)

STATUS = {"good": "#008300", "warning": "#eda100",
          "serious": "#eb6834", "critical": "#e34948"}


def use_style() -> None:
    """Apply the house style to matplotlib globally."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 160,
        "font.family": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        "figure.dpi": 110,
    })


def title(ax, headline: str, sub: str | None = None) -> None:
    """Headline states the finding; the subtitle carries the caveat/units."""
    ax.set_title(headline, loc="left", pad=14 if sub else 10)
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, color=INK_2)


def source(fig, text: str) -> None:
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=8, color=INK_MUTED)


def bar_ends(ax, horizontal: bool = False, radius_px: float = 3.5) -> None:
    """
    Round the corners of every bar by a fixed number of screen pixels.

    Naive rounding sets the corner radius in data units, which explodes when
    the two axes have wildly different scales - a price axis spanning 20 units
    beside a category axis spanning 7 produces spikes instead of corners. The
    radius here is converted from pixels into x-data units and the distortion
    is undone with `mutation_aspect`, so corners look the same on every chart.

    Call this after the axis limits are final; it reads them.
    """
    from matplotlib.patches import FancyBboxPatch

    ax.figure.canvas.draw_idle()
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    dx_per_px = (x1 - x0) / max(box.width, 1.0)
    dy_per_px = (y1 - y0) / max(box.height, 1.0)
    if dx_per_px <= 0 or dy_per_px <= 0:
        return
    aspect = dy_per_px / dx_per_px

    for p in list(ax.patches):
        if not hasattr(p, "get_width") or getattr(p, "_rounded", False):
            continue
        x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
        if w == 0 or h == 0:
            continue
        # never let the radius exceed a third of the bar's short side
        r = min(radius_px * dx_per_px, abs(w) / 3, abs(h) / (3 * aspect))
        if r <= 0:
            continue
        p.set_visible(False)
        new = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=0, facecolor=p.get_facecolor(), alpha=p.get_alpha(),
            mutation_aspect=aspect, zorder=p.get_zorder(),
        )
        new._rounded = True
        ax.add_patch(new)


def pct(ax, axis: str = "y", decimals: int = 0) -> None:
    from matplotlib.ticker import PercentFormatter
    f = PercentFormatter(xmax=1, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def money(ax, axis: str = "y", symbol: str = "£") -> None:
    from matplotlib.ticker import FuncFormatter

    def _f(v, _):
        a = abs(v)
        if a >= 1e6:
            return f"{symbol}{v/1e6:,.1f}M"
        if a >= 1e3:
            return f"{symbol}{v/1e3:,.0f}k"
        return f"{symbol}{v:,.0f}"
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(_f))


def save(fig, path) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {p}")
