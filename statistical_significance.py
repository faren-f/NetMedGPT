"""
Paired t-test comparing NetMedGPT vs TxGNN across seeds.

Splits with 5 seeds on both sides (suitable for paired t-test):
  - random_link_split  → TxGNN "random"
  - zero_shot_split    → TxGNN "complex_disease"

Disease-area checkpoints only have seed 1 for NetMedGPT, so we report
descriptive stats only (no paired test) for those.

Output: data/result/statistical_significance.csv
"""

import pandas as pd
import numpy as np
from scipy import stats

def bootstrap_ci(nm_vals, txg_vals, n_boot=10000, ci=95, seed=42):
    """Bootstrap CI on the mean paired difference (NetMedGPT - TxGNN)."""
    rng = np.random.default_rng(seed)
    diffs = nm_vals - txg_vals
    boot_means = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = (100 - ci) / 2
    hi = 100 - lo
    return float(np.percentile(boot_means, lo)), float(np.percentile(boot_means, hi))

# ── Load data ─────────────────────────────────────────────────────────────────
nm = pd.read_csv("data/result/all_checkpoints_evaluation.csv")
txgnn_all = pd.read_csv(
    "/home/bbc8731/txgnn_docker/TxGNN/reproduce/result_more_metrics.csv"
)
txgnn = txgnn_all[txgnn_all["Method"] == "TxGNN"].copy()

# Metric name mapping: NetMedGPT → TxGNN
METRIC_MAP = {
    "auc":          "AUROC",
    "auprc":        "AUPRC",
    "recall@100":   "Recall@100",
    "precision@100":"Precision@100",
}

# Split mapping: NetMedGPT inference → TxGNN Split
SPLIT_MAP = {
    "random_link_split": "random",
    "zero_shot_split":   "complex_disease",
}

EDGE_TYPES = ["indication", "contraindication"]

rows = []

# ── Paired t-test for random & zero-shot (5 seeds each) ──────────────────────
for nm_inference, txgnn_split in SPLIT_MAP.items():
    nm_sub = nm[
        (nm["inference"] == nm_inference) &
        (nm["edge_type"].isin(EDGE_TYPES))
    ].copy()

    txgnn_sub = txgnn[txgnn["Split"] == txgnn_split].copy()

    for edge in EDGE_TYPES:
        nm_e  = nm_sub[nm_sub["edge_type"] == edge].sort_values("seed")
        txg_e = txgnn_sub[txgnn_sub["Task"] == edge]

        for nm_metric, txgnn_metric in METRIC_MAP.items():
            nm_vals = nm_e[nm_metric].values  # shape (5,)

            txg_vals = (
                txg_e[txg_e["Metric Name"] == txgnn_metric]
                .sort_values("Seed")["Metric"]
                .values  # shape (5,)
            )

            if len(nm_vals) != 5 or len(txg_vals) != 5:
                print(f"WARNING: missing data for {nm_inference}/{edge}/{nm_metric} "
                      f"(nm={len(nm_vals)}, txgnn={len(txg_vals)})")
                continue

            t_stat, p_val = stats.ttest_rel(nm_vals, txg_vals)
            mean_diff = float(np.mean(nm_vals - txg_vals))
            nm_mean   = float(np.mean(nm_vals))
            txg_mean  = float(np.mean(txg_vals))
            nm_std    = float(np.std(nm_vals, ddof=1))
            txg_std   = float(np.std(txg_vals, ddof=1))
            ci_lo, ci_hi = bootstrap_ci(nm_vals, txg_vals)

            rows.append({
                "split":          nm_inference,
                "edge_type":      edge,
                "metric":         nm_metric,
                "netmedgpt_mean": round(nm_mean,  4),
                "netmedgpt_std":  round(nm_std,   4),
                "txgnn_mean":     round(txg_mean, 4),
                "txgnn_std":      round(txg_std,  4),
                "mean_diff":      round(mean_diff, 4),
                "ci_low":         round(ci_lo,    4),
                "ci_high":        round(ci_hi,    4),
                "t_stat":         round(t_stat,   4),
                "p_value":        round(p_val,    4),
                "significant":    p_val < 0.05,
            })

# ── Disease-area: descriptive only (1 seed for NetMedGPT) ────────────────────
DISEASE_AREAS = [
    "adrenal_gland", "anemia", "autoimmune", "cardiovascular",
    "cell_proliferation", "diabetes", "mental_health",
    "metabolic_disorder", "neurodigenerative",
]

nm_da = nm[nm["inference"].isin(DISEASE_AREAS) & nm["edge_type"].isin(EDGE_TYPES)]

for area in DISEASE_AREAS:
    txg_sub = txgnn[txgnn["Split"] == area]
    nm_sub  = nm_da[nm_da["inference"] == area]

    for edge in EDGE_TYPES:
        nm_e  = nm_sub[nm_sub["edge_type"] == edge]
        txg_e = txg_sub[txg_sub["Task"] == edge]

        for nm_metric, txgnn_metric in METRIC_MAP.items():
            nm_val  = nm_e[nm_metric].values
            txg_vals = (
                txg_e[txg_e["Metric Name"] == txgnn_metric]
                .sort_values("Seed")["Metric"].values
            )

            if len(nm_val) == 0 or len(txg_vals) == 0:
                continue

            nm_mean  = float(nm_val[0])
            txg_mean = float(np.mean(txg_vals))
            txg_std  = float(np.std(txg_vals, ddof=1))

            rows.append({
                "split":          area,
                "edge_type":      edge,
                "metric":         nm_metric,
                "netmedgpt_mean": round(nm_mean,  4),
                "netmedgpt_std":  float("nan"),
                "txgnn_mean":     round(txg_mean, 4),
                "txgnn_std":      round(txg_std,  4),
                "mean_diff":      round(nm_mean - txg_mean, 4),
                "ci_low":         float("nan"),
                "ci_high":        float("nan"),
                "t_stat":         float("nan"),
                "p_value":        float("nan"),
                "significant":    float("nan"),
            })

# ── Save & print ──────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
out = "data/result/statistical_significance.csv"
df.to_csv(out, index=False)
print(f"Saved to {out}\n")

# Print paired t-test summary
paired = df[df["split"].isin(SPLIT_MAP.keys())]
print("=" * 110)
print("PAIRED T-TEST + BOOTSTRAP 95% CI: NetMedGPT vs TxGNN (5 seeds)")
print("=" * 110)
print(f"{'Split':<22} {'Edge':<18} {'Metric':<15} {'NM mean':>9} {'TxGNN mean':>11} "
      f"{'Diff':>8} {'95% CI':>18} {'p-value':>9} {'Sig?':>6}")
print("-" * 110)
for _, r in paired.sort_values(["split","edge_type","metric"]).iterrows():
    sig = "*" if r["significant"] else ""
    ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
    print(f"{r['split']:<22} {r['edge_type']:<18} {r['metric']:<15} "
          f"{r['netmedgpt_mean']:>9.4f} {r['txgnn_mean']:>11.4f} "
          f"{r['mean_diff']:>8.4f} {ci_str:>18} {r['p_value']:>9.4f} {sig:>6}")

# Print disease-area summary
disease = df[df["split"].isin(DISEASE_AREAS)]
print("\n" + "=" * 90)
print("DISEASE-AREA: NetMedGPT (seed 1) vs TxGNN mean ± std (5 seeds) — descriptive only")
print("=" * 90)
print(f"{'Split':<22} {'Edge':<18} {'Metric':<15} {'NM':>9} {'TxGNN mean':>11} "
      f"{'TxGNN std':>10} {'Diff':>8}")
print("-" * 90)
for _, r in disease.sort_values(["split","edge_type","metric"]).iterrows():
    print(f"{r['split']:<22} {r['edge_type']:<18} {r['metric']:<15} "
          f"{r['netmedgpt_mean']:>9.4f} {r['txgnn_mean']:>11.4f} "
          f"{r['txgnn_std']:>10.4f} {r['mean_diff']:>8.4f}")

# ── AUPRC focused summary: random_link_split and zero_shot_split ──────────────
print("\n" + "=" * 100)
print("AUPRC SUMMARY: NetMedGPT vs TxGNN — random_link_split & zero_shot_split (5 seeds)")
print("=" * 100)
print(f"{'Split':<22} {'Edge':<18} {'NM seeds':>36} {'NM mean':>8} {'TxGNN mean':>11} "
      f"{'Diff':>7} {'95% CI':>18} {'p-value':>9}")
print("-" * 100)

txgnn_auprc = txgnn[txgnn["Metric Name"] == "AUPRC"]
for nm_split, txg_split in SPLIT_MAP.items():
    for edge in EDGE_TYPES:
        nm_vals  = nm[(nm["inference"] == nm_split) & (nm["edge_type"] == edge)].sort_values("seed")["auprc"].values
        txg_vals = txgnn_auprc[(txgnn_auprc["Split"] == txg_split) & (txgnn_auprc["Task"] == edge)].sort_values("Seed")["Metric"].values
        _, p_val = stats.ttest_rel(nm_vals, txg_vals)
        ci_lo, ci_hi = bootstrap_ci(nm_vals, txg_vals)
        seeds_str = str(nm_vals.round(4).tolist())
        print(f"{nm_split:<22} {edge:<18} {seeds_str:>36} {nm_vals.mean():>8.4f} "
              f"{txg_vals.mean():>11.4f} {nm_vals.mean()-txg_vals.mean():>7.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}] {p_val:>9.4f}")
