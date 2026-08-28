#!/usr/bin/env python3
"""
Offline DECIDER: does the ConvLSTM (temporal) arm earn its place?

Zero training compute. Re-ranks the already-logged per-client per-round scores:
  stat_ewma  (statistical arm)  vs  convlstm_rel (temporal arm)  vs  fused_score
against ground-truth is_malicious, on the real MNIST + Fashion-MNIST July runs.

Two views, both threshold-honest:
  (A) Threshold-free separation:  ROC-AUC and Average Precision per arm.
  (B) Matched-budget operating point:  per round r>=6, remove the SAME number of
      clients K_r the fused detector actually removed (from predict_state), but
      chosen by each arm's own score -> recall and benign-rejection per arm.
      This isolates "given a fixed removal budget, which arm picks the right ones?"
"""
import os, re, glob, math
import numpy as np
import pandas as pd

OUT = "/home/tuan/get-local-model-bt-id/rl-flwr/outputs"
WARMUP_LAST = 5          # malicious active from round 6
PAT = re.compile(r'^\d\d-\d\d-\d\d_(mnist|fashion_mnist|cifar10)_(gaussian|lie|min_max|no_attack)_tau([0-9.]+)$')

def auc_roc(y, s):
    """Rank-based ROC-AUC (Mann-Whitney), tie-safe. y in {0,1}."""
    y = np.asarray(y); s = np.asarray(s, float)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sr = s[order]
    i = 0; r = 1
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j+1] == sr[i]:
            j += 1
        ranks[order[i:j+1]] = (r + (r + (j - i))) / 2.0
        r += (j - i + 1); i = j + 1
    return (ranks[y == 1].sum() - npos*(npos+1)/2.0) / (npos*nneg)

def avg_precision(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    npos = int(y.sum())
    if npos == 0: return np.nan
    order = np.argsort(-s, kind="mergesort")
    yo = y[order]
    tp = np.cumsum(yo)
    prec = tp / (np.arange(len(yo)) + 1.0)
    return float((prec * yo).sum() / npos)

def matched_budget(df):
    """df: rows for one run, r>=6, with is_malicious, predict_state, arm scores.
       Per round, remove top-K_r by each arm; K_r = #predict_state==1 that round."""
    arms = ["stat_ewma", "convlstm_rel", "fused_score"]
    agg = {a: {"tp":0,"fp":0} for a in arms}
    agg["fused_actual"] = {"tp":0,"fp":0}
    n_mal = int((df.is_malicious==1).sum()); n_ben = int((df.is_malicious==0).sum())
    for r, g in df.groupby("round"):
        K = int((g.predict_state==1).sum())
        mal = g.is_malicious.values
        # actual fused decision as executed
        pa = g.predict_state.values.astype(bool)
        agg["fused_actual"]["tp"] += int((mal[pa]==1).sum())
        agg["fused_actual"]["fp"] += int((mal[pa]==0).sum())
        if K <= 0: continue
        for a in arms:
            idx = np.argsort(-g[a].values, kind="mergesort")[:K]
            sel = np.zeros(len(g), bool); sel[idx] = True
            agg[a]["tp"] += int((mal[sel]==1).sum())
            agg[a]["fp"] += int((mal[sel]==0).sum())
    out = {}
    for k,v in agg.items():
        out[f"recall_{k}"] = v["tp"]/n_mal if n_mal else np.nan
        out[f"benrej_{k}"] = v["fp"]/n_ben if n_ben else np.nan
    return out

rows = []
for d in sorted(glob.glob(os.path.join(OUT, "2026-07-*", "*"))):
    name = os.path.basename(d)
    m = PAT.match(name)
    if not m: continue
    ds, atk, qs = m.group(1), m.group(2), float(m.group(3))
    if ds == "cifar10" or atk == "no_attack": continue
    sa = os.path.join(d, "stat_anomaly.csv")
    cs = os.path.join(d, "client_states_rounds.csv")
    if not os.path.exists(sa): continue
    df = pd.read_csv(sa)
    if os.path.exists(cs):
        cst = pd.read_csv(cs)[["round","client_id","predict_state"]]
        df = df.merge(cst, on=["round","client_id"], how="left")
    else:
        df["predict_state"] = np.nan
    g = df[df["round"] > WARMUP_LAST].copy()
    if g.is_malicious.nunique() < 2: continue
    rec = dict(dataset=ds, attack=atk, q_s=qs,
               auc_stat=auc_roc(g.is_malicious, g.stat_ewma),
               auc_convlstm=auc_roc(g.is_malicious, g.convlstm_rel),
               auc_fused=auc_roc(g.is_malicious, g.fused_score),
               ap_stat=avg_precision(g.is_malicious, g.stat_ewma),
               ap_convlstm=avg_precision(g.is_malicious, g.convlstm_rel),
               ap_fused=avg_precision(g.is_malicious, g.fused_score))
    gp = g.dropna(subset=["predict_state"]) if "predict_state" in g else g.iloc[0:0]
    if len(gp) and gp.is_malicious.nunique() >= 2:
        rec.update(matched_budget(gp))
    rows.append(rec)

R = pd.DataFrame(rows).sort_values(["dataset","attack","q_s"]).reset_index(drop=True)
R.to_csv("/tmp/claude-1000/-home-tuan-paper1-convlstm-ano/da389f53-7ad2-4b94-83d4-9ac03d540f85/scratchpad/decider_results.csv", index=False)
print(f"runs analyzed: {len(R)}   (expect 66 = 2 datasets x 3 attacks x 11 q_s)\n")

pd.set_option("display.width", 200, "display.max_columns", 40, "display.float_format", lambda x: f"{x:.3f}")

print("="*78)
print("VIEW A - Threshold-free separation (ROC-AUC), mean over the 11 q_s values")
print("  Reads: can each arm rank malicious above benign?  1.0 = perfect, 0.5 = none")
print("="*78)
print(R.groupby(["dataset","attack"])[["auc_stat","auc_convlstm","auc_fused"]].mean().to_string())

print("\n" + "="*78)
print("VIEW A' - AUC at the two extremes: q_s=0.0 (temporal arm FRESH) vs q_s=1.0 (STARVED)")
print("="*78)
for qs in (0.0, 1.0):
    sub = R[np.isclose(R.q_s, qs)]
    print(f"\n-- q_s = {qs:.1f} --")
    print(sub.set_index(["dataset","attack"])[["auc_stat","auc_convlstm","auc_fused"]].to_string())

if "recall_stat" in R.columns:
    print("\n" + "="*78)
    print("VIEW B - Matched-budget operating point (mean over q_s): recall | benign-rejection")
    print("  Same per-round removal count for every arm; which arm picks the right clients?")
    print("="*78)
    cols = ["recall_stat","recall_convlstm","recall_fused_actual",
            "benrej_stat","benrej_convlstm","benrej_fused_actual"]
    have = [c for c in cols if c in R.columns]
    print(R.groupby(["dataset","attack"])[have].mean().to_string())

print("\n" + "="*78)
print("DECIDER SUMMARY")
print("="*78)
d = R.groupby(["dataset","attack"])[["auc_stat","auc_convlstm","auc_fused"]].mean()
d["fused_minus_stat"] = d.auc_fused - d.auc_stat
d["convlstm_vs_stat"] = d.auc_convlstm - d.auc_stat
print(d[["fused_minus_stat","convlstm_vs_stat"]].to_string())
print(f"\nMean AUC gain from fusion over stat-only : {d.fused_minus_stat.mean():+.4f}")
print(f"Max  AUC gain from fusion over stat-only : {d.fused_minus_stat.max():+.4f}  "
      f"({d.fused_minus_stat.idxmax()})")
print(f"Configs where temporal-only beats stat-only (AUC): "
      f"{int((d.convlstm_vs_stat>0).sum())} of {len(d)}")
