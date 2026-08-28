#!/usr/bin/env python3
"""make_tables.py -- regenerate the paper's results tables from the run outputs.

Provenance rules this script enforces, each of which has burned us before:
  * The run DIRECTORY NAME is the only provenance. run_config.json records the
    sweep's *defaults*, not the cell, so it mislabels every point.
  * Detection metrics are computed over the ATTACKED rounds t >= 6 only. The
    summary CSV's `fdr` column pools the five warm-up rounds, in which no
    adversary is active and every exclusion is honest by construction; pooling
    inflates the methods unequally and reverses head-to-head verdicts.
  * Only runs that reached the requested horizon are eligible. A 30-round run
    and a 50-round run are not comparable and are never placed in one table.
  * Where a cell has several runs, the LATEST is taken.
"""
from __future__ import annotations
import csv, glob, json, os, re, collections, sys

HORIZON  = int(os.environ.get("HORIZON", "50"))
NCLIENTS = int(os.environ.get("NCLIENTS", "50"))
ROOT = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r"^(?P<t>\d\d-\d\d-\d\d)_(?P<ds>mnist|fashion_mnist|cifar10)_"
                 r"(?P<def>.+?)_(?P<atk>gaussian|lie|min_max|no_attack|fang)_tau(?P<tau>[\d.]+)$")

def collect():
    best = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "outputs/2026-*/*/"))):
        m = PAT.match(os.path.basename(d.rstrip("/")))
        if not m:
            continue
        cs, rj = os.path.join(d, "client_states_rounds.csv"), os.path.join(d, "results.json")
        if not (os.path.exists(cs) and os.path.exists(rj)):
            continue
        try:
            res = json.load(open(rj))
        except Exception:
            continue
        if not isinstance(res, list) or not res:
            continue
        R = max(r["round"] for r in res)
        if R != HORIZON:                      # never mix horizons
            continue
        acc = [r for r in res if r["round"] == R][0]["evaluate_metrics_serverapp"]["accuracy"]
        per = collections.defaultdict(lambda: [0, 0, 0])   # TP, FP, FN
        nseen = set()
        for r in csv.DictReader(open(cs)):
            rd = int(r["round"]); nseen.add(r["client_id"])
            mal, pred = r["is_malicious"] == "1", r["predict_state"] == "1"
            k = per[rd]
            k[0] += mal and pred; k[1] += (not mal) and pred; k[2] += mal and not pred
        att = [v for rd, v in per.items() if rd >= 6]
        wu  = [v for rd, v in per.items() if rd < 6]
        if not att:
            continue
        # Federation size is NOT recorded anywhere but the client roster, and the
        # 2026-08-1x campaign ran the same 50-round horizon at N=100. Without this
        # filter those rows silently land in an N=50 table -- Multi-Krum dropping
        # 40 per round out of a 50-client federation is what exposed it.
        if len(nseen) != NCLIENTS:
            continue
        TP = sum(v[0] for v in att); FP = sum(v[1] for v in att); FN = sum(v[2] for v in att)
        M = TP + FN
        row = dict(ds=m["ds"], dfn=m["def"], atk=m["atk"], tau=float(m["tau"]), R=R, N=len(nseen),
                   acc=acc,
                   fdr=FP / (TP + FP) if TP + FP else 0.0,
                   recall=TP / M if M else float("nan"),
                   drop=(TP + FP) / len(att),
                   warm=sum(v[0] + v[1] for v in wu) / len(wu) if wu else float("nan"),
                   stamp=d.split("/")[-3] + " " + m["t"], dir=os.path.basename(d.rstrip("/")))
        # ASR := mean over attacked rounds of (undetected attackers / federation size).
        # It saturates at M/N = 0.40 when the defense detects nobody, NOT at 1.0 --
        # read the ceiling as 0.40 in every table below.
        row["asr"] = sum(v[2] for v in att) / len(att) / max(1, len(nseen))
        key = (m["ds"], m["def"], m["atk"], float(m["tau"]))
        if key not in best or row["stamp"] > best[key]["stamp"]:
            best[key] = row
    return best

DS    = ["mnist", "fashion_mnist"]
ATK   = ["gaussian", "lie", "min_max", "fang"]
ROWS  = [("fedavg", "FedAvg (no defense)", ""), ("fldetector", "FLDetector", r"$^{\S}$"),
         ("fltrust", "FLTrust", r"$^{\P}$"), ("multikrum", "Multi-Krum", r"$^\dagger$"),
         ("dnc", "DnC", r"$^\dagger$"), ("ustad_cal", r"\textbf{USTAD-FL (ours)}", "")]
TAU   = float(os.environ.get("TAU", "0.6"))

def cell(best, ds, dfn, atk, field, fmt="{:.3f}"):
    r = best.get((ds, dfn, atk, TAU))
    return fmt.format(r[field]) if r else r"\pending"

def main():
    best = collect()
    have = collections.Counter(k[1] for k in best)
    print("%% N=%d  horizon=%d  tau=%.2f  cells per defense: %s"
          % (NCLIENTS, HORIZON, TAU, dict(have)), file=sys.stderr)
    out = []
    for key, field, fmt in [("acc", "acc", "{:.3f}"), ("asr", "asr", "{:.3f}"),
                            ("recall", "recall", "{:.3f}"), ("fdr", "fdr", "{:.3f}"),
                            ("drop", "drop", "{:.1f}"), ("warm", "warm", "{:.1f}")]:
        out.append("%% ---- %s ----" % key)
        for dfn, label, dag in ROWS:
            cells = [cell(best, ds, dfn, a, field, fmt) for ds in DS for a in ATK]
            out.append("%s%s & %s \\\\" % (label, dag, " & ".join(cells)))
    print("\n".join(out))

if __name__ == "__main__" and os.environ.get("EMIT") != "1":
    main()

# ---------------------------------------------------------------------------
def emit_tables():
    """Emit the two complete table environments for fl_sec.tex."""
    best = collect()
    def C(ds, dfn, atk, f, fmt="{:.3f}"):
        return cell(best, ds, dfn, atk, f, fmt)
    def block(field, fmt="{:.3f}", rows=ROWS):
        return "\n".join(
            "  & %s%s & %s \\\\" % (lab, dag, " & ".join(
                C(ds, d, a, field, fmt) for ds in DS for a in ATK))
            for d, lab, dag in rows)

    t1 = r"""\begin{table*}[!t]
\centering
\caption{Final global test accuracy at round~$50$ under $40\%$ Byzantine clients ($N=50$, $M=20$), including the aggregator-targeted Fang attack.}
\label{tab:main_results}
\footnotesize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcccccccc@{}}
\toprule
& \multicolumn{4}{c}{\textbf{MNIST}} & \multicolumn{4}{c}{\textbf{Fashion-MNIST}} \\
\cmidrule(lr){2-5} \cmidrule(l){6-9}
\textbf{Method} & Gauss. & LIE & Min-Max & \textbf{Fang} & Gauss. & LIE & Min-Max & \textbf{Fang} \\
\midrule
""" + "\n".join(
        "%s%s & %s \\\\" % (lab, dag, " & ".join(
            C(ds, d, a, "acc") for ds in DS for a in ATK))
        for d, lab, dag in ROWS) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
{\footnotesize \raggedright All rows share one protocol: $50$ rounds, $10$ local epochs, $N=50$ clients of which $M=20$ are malicious, non-IID Dirichlet $\alpha=0.5$, adversary dormant until round~$6$, single run per cell. The Fang columns are the axis this revision adds and they reorder the field: the three aggregator-agnostic attacks separate the defenses only coarsely, whereas Fang separates them decisively and in the opposite direction. $^\dagger$Multi-Krum and DnC are each granted the exact malicious count $M$ and exclude exactly that many participants every round; no other method here, ours included, is given it. $^{\S}$FLDetector begins detecting at iteration~$50$, following its own paper, which trains for $1{,}000$; at our horizon its detector never engages, so its row measures undefended FedAvg. Its high Fang figure is therefore \emph{not} a defensive result---it is undefended FedAvg, which Fang barely harms. $^{\P}$FLTrust is markedly unstable across runs and single cells should not be read as precise.\par}
\end{table*}"""

    t2 = r"""\begin{table*}[!t]
\centering
\caption{Detection behaviour over the attacked rounds $t \ge 6$, and unconditional exclusions on the adversary-free warm-up rounds ($N=50$, $M=20$).}
\label{tab:detection_results}
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{@{}llcccccccc@{}}
\toprule
& & \multicolumn{4}{c}{\textbf{MNIST}} & \multicolumn{4}{c}{\textbf{Fashion-MNIST}} \\
\cmidrule(lr){3-6} \cmidrule(l){7-10}
\textbf{Metric} & \textbf{Method} & Gauss. & LIE & Min-Max & \textbf{Fang} & Gauss. & LIE & Min-Max & \textbf{Fang} \\
\midrule
\multirow{6}{*}{Recall ($t\ge6$)}
""" + block("recall") + r"""
\midrule
\multirow{6}{*}{FDR ($t\ge6$)}
""" + block("fdr") + r"""
\midrule
\multirow{6}{*}{Excl./round ($t\ge6$)}
""" + block("drop", "{:.1f}") + r"""
\midrule
\multirow{6}{*}{Excl./round (warm-up)}
""" + block("warm", "{:.1f}") + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
{\footnotesize \raggedright Recall and FDR are computed over the attacked rounds $t \ge 6$ only. Pooling the five warm-up rounds---in which no adversary is active, so every exclusion is honest by construction---inflates the methods unequally and reverses head-to-head verdicts; we therefore never report the pooled figure. The final block is the measurement that matters for Section~\ref{sec:drop_floor}: exclusions per round on rounds $1$--$5$, where the adversary is dormant and \emph{every} excluded client is honest. It is the direct measurement of an unconditional drop rate, and USTAD-FL's entry in it is a cost of our design, not a result in its favour. $^\dagger$Multi-Krum and DnC retain a fixed count derived from a known $M$, so their attacked-round FDR is arithmetic rather than discrimination; the warm-up block shows the same fixed count falling entirely on honest clients. $^{\S}$FLDetector's zeros are those of a detector that never engaged, not of a clean one.\par}
\end{table*}"""
    return t1, t2

if __name__ == "__main__" and os.environ.get("EMIT") == "1":
    a, b = emit_tables(); print(a); print(); print(b)
