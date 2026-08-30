"""Shared IEEE figure style for this project's plots.

Encodes the figure rules the manuscript is held to:

  H.01  Times New Roman (or a metric-compatible substitute) everywhere --
        axis labels, tick labels, legend text, annotations -- so figure text
        matches the IEEEtran body font.
  H.02  8 pt legend text, matching \\footnotesize.
  H.03  Legend inside the axes, in the least dense corner.
  H.06  First x tick sits at the axis start; no gap before the first point.
  H.07  y axis starts at 0 or at the lowest datum; no padding underneath.
  D.01  NO title inside the figure. IEEE captions sit below the figure and are
        set by \\caption{} in LaTeX, so an embedded title duplicates it and is
        typeset in the wrong font at the wrong size. Every plotting script here
        emits a caption string for the author to paste instead.

Column widths are the IEEEtran defaults: 3.5 in single column, 7.16 in double.
"""

from __future__ import annotations

import glob
import matplotlib
from matplotlib import font_manager

COL_SINGLE = 3.5      # in, IEEEtran \columnwidth
COL_DOUBLE = 7.16     # in, IEEEtran \textwidth

# Ink and rule colours; series colours are chosen per figure.
INK, INK_2, GRID = "#000000", "#3f3f3f", "#c8c8c8"


def _register_serif() -> str:
    """Return the best available Times-metric serif, registering it if needed.

    Times New Roman is not redistributable on Linux; Liberation Serif and
    Nimbus Roman are metric-compatible substitutes and are what IEEE submissions
    from Linux normally embed. Fall back to matplotlib's bundled serif so a
    missing font degrades the typeface rather than crashing the build.
    """
    for pattern, name in (("/usr/share/fonts/**/LiberationSerif-*.ttf", "Liberation Serif"),
                          ("/usr/share/fonts/**/NimbusRoman-*.otf", "Nimbus Roman"),
                          ("/usr/share/fonts/**/times*.ttf", "Times New Roman")):
        files = glob.glob(pattern, recursive=True)
        if files:
            for f in files:
                try:
                    font_manager.fontManager.addfont(f)
                except Exception:
                    pass
            return name
    return "DejaVu Serif"


SERIF = _register_serif()


def apply(base: float = 8.0) -> None:
    """Apply the IEEE figure rcParams. `base` is the body text size in points."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": [SERIF, "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",          # Times-like math to match the text
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base,             # H.02: 8 pt
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.5,
        "grid.color": GRID,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 600,                  # raster fallback; prefer PDF
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,                  # embed TrueType, not Type 3
        "ps.fonttype": 42,
    })


def tighten_x(ax, x) -> None:
    """H.06 -- first tick at the axis start, last at the axis end."""
    ax.set_xticks(x)
    ax.set_xlim(min(x), max(x))


def tighten_y(ax, y, bottom=0.0) -> None:
    """H.07 -- no whitespace below the data; start at 0 or the lowest datum."""
    ax.set_ylim(bottom=bottom, top=max(y) * 1.02, auto=False)
