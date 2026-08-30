"""Per-measure q_s sweep panels, one file each, for \\subfloat assembly.

H.05 requires multi-panel IEEE figures to be built from SEPARATE single-panel
files combined with \\subfloat in LaTeX, not emitted as one composite image, so
that each panel scales independently and its subcaption is typeset by LaTeX in
the body font. This script therefore writes one file per measure and prints the
\\subfloat block to paste.

Measures, against the scoring quantile q_s:
  (a) ConvLSTM forward passes      -- detector cost
  (b) worst-cell FDR, rounds >= 6  -- the quantity that selects the operating point
  (c) final accuracy               -- flat, and therefore cannot select it

Usage:
    python plot_sweep_tau_panels.py                       # default 66-run sweep
    python plot_sweep_tau_panels.py --dataset mnist
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ieee_style

ROOT = Path(__file__).parent
C_COST, C_FDR, C_ACC = "#2a78d6", "#eb6834", "#1baf7a"
SEL = 0.4


def load():
    spec = importlib.util.spec_from_file_location("mt", ROOT / "make_tables.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.cost_rows(), m.COST_N * m.COST_T


def panel(x, y, color, ylabel, out, sel, ybottom=0.0, hline=None):
    ieee_style.apply(base=8.0)
    fig, ax = plt.subplots(figsize=(ieee_style.COL_SINGLE * 0.49,
                                    ieee_style.COL_SINGLE * 0.42))
    ax.plot(x, y, marker="o", color=color, zorder=3)
    ax.set_xlabel("$q_s$")
    ax.set_ylabel(ylabel)
    ieee_style.tighten_x(ax, [t for i, t in enumerate(x) if i % 2 == 0])  # H.06
    ax.set_xlim(min(x), max(x))
    ieee_style.tighten_y(ax, y, bottom=ybottom)                          # H.07
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    if hline is not None:
        ax.axhline(hline, color=ieee_style.INK_2, linewidth=0.6,
                   linestyle=(0, (1.5, 2)), zorder=1)
        ax.set_ylim(ybottom, hline, auto=False)
    if sel in x:
        ax.axvline(sel, color=ieee_style.INK_2, linewidth=0.6, zorder=1)
    fig.savefig(out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", type=float, default=SEL)
    ap.add_argument("--stem", default="fig_qs")
    args = ap.parse_args()

    rs, exhaustive = load()
    x = [r["tau"] for r in rs]
    files = []
    for suffix, y, color, ylabel, bottom, hline in [
        ("a", [r["fwd"] for r in rs], C_COST, "ConvLSTM forwards", 0.0, exhaustive),
        ("b", [r["fdr_worst"] for r in rs], C_FDR, "Worst-cell FDR", 0.0, None),
        ("c", [r["acc"] for r in rs], C_ACC, "Final accuracy", 0.0, 1.0),
    ]:
        out = ROOT / f"{args.stem}_{suffix}.pdf"
        panel(x, y, color, ylabel, out, args.select, bottom, hline)
        files.append(out.name)

    print("\nLaTeX (H.04/H.05: subcaptions below each panel, caption below the figure):\n")
    subs = [("Detector cost.", files[0]), ("Worst-cell FDR ($t\\ge6$).", files[1]),
            ("Final accuracy.", files[2])]
    print("\\begin{figure}[t]\n  \\centering")
    for i, (cap, f) in enumerate(subs):
        print(f"  \\subfloat[{cap}]{{%\n    "
              f"\\includegraphics[width=0.32\\columnwidth]{{{f}}}%\n    "
              f"\\label{{fig:qs_{chr(97+i)}}}%\n  }}"
              + ("\n  \\hfill" if i < len(subs) - 1 else ""))
    print("  \\caption{Operating-point sweep over the scoring quantile $q_s$. Detector "
          "cost falls monotonically while worst-cell false discovery does not, being "
          "minimized in a narrow band near $q_s=0.4$; accuracy is flat throughout and "
          "therefore cannot select the operating point.}")
    print("  \\label{fig:qs_sweep_panels}\n\\end{figure}")


if __name__ == "__main__":
    main()
