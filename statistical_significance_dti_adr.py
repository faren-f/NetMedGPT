"""
Statistical significance tests for DTI, ADR, and disease-area tasks.

DTI          : NetMedGPT vs GraphormerDTI (5 seeds, 1,667-protein restricted pool)
               Data: GraphormerDTI/comparison_results/drug_protein_per_seed_5seeds.csv

ADR          : NetMedGPT vs MFAHGN (5 seeds, 990-effect restricted pool)
               Data: MFAHGN/ADR_comparison/results/netmedgpt_adr_restricted_pool.json
                     MFAHGN/ADR_comparison/results/seed{1-5}_mfahgn_results.json

Disease area : One-sample t-test (NetMedGPT seed 1 vs TxGNN 5-seed distribution).
               NetMedGPT has only 1 seed per area so paired t-test is not possible.
               Test statistic: t = (nm - txgnn_mean) / (txgnn_std / sqrt(5)), df=4.

Both use paired t-test + bootstrap 95% CI (10,000 resamples).
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

DTI_CSV     = "/home/bbc8731/GraphormerDTI/comparison_results/drug_protein_per_seed_5seeds.csv"
NM_ADR_JSON = "/home/bbc8731/MFAHGN/ADR_comparison/results/netmedgpt_adr_restricted_pool.json"
MFAHGN_DIR  = "/home/bbc8731/MFAHGN/ADR_comparison/results"


def bootstrap_ci(nm_vals, comp_vals, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    diffs = nm_vals - comp_vals
    boot_means = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return float(np.percentile(boot_means, lo)), float(np.percentile(boot_means, hi))


def test_pair(nm_vals, comp_vals, metric, nm_name="NetMedGPT", comp_name="Competitor"):
    t_stat, p_val = stats.ttest_rel(nm_vals, comp_vals)
    mean_diff = float(np.mean(nm_vals - comp_vals))
    ci_lo, ci_hi = bootstrap_ci(nm_vals, comp_vals)
    return {
        "metric":        metric,
        "nm_mean":       round(float(np.mean(nm_vals)),  4),
        "nm_std":        round(float(np.std(nm_vals, ddof=1)), 4),
        "comp_mean":     round(float(np.mean(comp_vals)), 4),
        "comp_std":      round(float(np.std(comp_vals, ddof=1)), 4),
        "mean_diff":     round(mean_diff, 4),
        "ci_low":        round(ci_lo, 4),
        "ci_high":       round(ci_hi, 4),
        "t_stat":        round(float(t_stat), 4),
        "p_value":       round(float(p_val), 4),
        "significant":   p_val < 0.05,
    }


# ── DTI: NetMedGPT vs GraphormerDTI ──────────────────────────────────────────
print("Loading DTI results...")
dti = pd.read_csv(DTI_CSV)
nm_dti   = dti[dti["model"] == "NetMedGPT"].sort_values("seed").reset_index(drop=True)
gdti     = dti[dti["model"] == "GraphormerDTI"].sort_values("seed").reset_index(drop=True)

dti_rows = []
for metric, nm_col, comp_col in [
    ("AUC",        "auc",       "auc"),
    ("AUPR",       "aupr",      "aupr"),
    ("Recall@100", "recall_100","recall_100"),
]:
    nm_vals   = nm_dti[nm_col].values
    comp_vals = gdti[comp_col].values
    row = test_pair(nm_vals, comp_vals, metric)
    dti_rows.append(row)

dti_df = pd.DataFrame(dti_rows)

print("\n" + "=" * 105)
print("DTI: NetMedGPT vs GraphormerDTI (5 seeds, 1,667-protein restricted pool)")
print("=" * 105)
print(f"{'Metric':<14} {'NM mean':>9} {'NM std':>8} {'GDTI mean':>10} {'GDTI std':>9} "
      f"{'Diff':>8} {'95% CI':>20} {'p-value':>9} {'Sig?':>6}")
print("-" * 105)
for _, r in dti_df.iterrows():
    sig = "*" if r["significant"] else ""
    ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
    print(f"{r['metric']:<14} {r['nm_mean']:>9.4f} {r['nm_std']:>8.4f} "
          f"{r['comp_mean']:>10.4f} {r['comp_std']:>9.4f} "
          f"{r['mean_diff']:>8.4f} {ci_str:>20} {r['p_value']:>9.4f} {sig:>6}")


# ── ADR: NetMedGPT vs MFAHGN ──────────────────────────────────────────────────
print("\nLoading ADR results...")
with open(NM_ADR_JSON) as f:
    nm_adr = json.load(f)

nm_seeds = {s["seed"]: s for s in nm_adr["seeds"]}

mfahgn_seeds = {}
for s in range(1, 6):
    with open(f"{MFAHGN_DIR}/seed{s}_mfahgn_results.json") as f:
        mfahgn_seeds[s] = json.load(f)

adr_rows = []
for metric, nm_key, mf_key in [
    ("AUC",        "auroc",      "auroc"),
    ("AUPR",       "aupr",       "aupr"),
    ("Recall@100", "recall_100", "recall_100"),
]:
    nm_vals   = np.array([nm_seeds[s][nm_key]   for s in range(1, 6)])
    comp_vals = np.array([mfahgn_seeds[s][mf_key] for s in range(1, 6)])
    row = test_pair(nm_vals, comp_vals, metric)
    adr_rows.append(row)

adr_df = pd.DataFrame(adr_rows)

print("\n" + "=" * 105)
print("ADR: NetMedGPT vs MFAHGN (5 seeds, 990-effect restricted pool)")
print("=" * 105)
print(f"{'Metric':<14} {'NM mean':>9} {'NM std':>8} {'MFAHGN mean':>12} {'MFAHGN std':>11} "
      f"{'Diff':>8} {'95% CI':>20} {'p-value':>9} {'Sig?':>6}")
print("-" * 105)
for _, r in adr_df.iterrows():
    sig = "*" if r["significant"] else ""
    ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
    print(f"{r['metric']:<14} {r['nm_mean']:>9.4f} {r['nm_std']:>8.4f} "
          f"{r['comp_mean']:>12.4f} {r['comp_std']:>11.4f} "
          f"{r['mean_diff']:>8.4f} {ci_str:>20} {r['p_value']:>9.4f} {sig:>6}")


# ── Save ──────────────────────────────────────────────────────────────────────
dti_df.insert(0, "task", "DTI")
dti_df.insert(1, "comparison", "NetMedGPT vs GraphormerDTI")
adr_df.insert(0, "task", "ADR")
adr_df.insert(1, "comparison", "NetMedGPT vs MFAHGN")

out = pd.concat([dti_df, adr_df], ignore_index=True)
out.to_csv("data/result/statistical_significance_dti_adr.csv", index=False)
print(f"\nSaved to data/result/statistical_significance_dti_adr.csv")


# ── Off-label use: NetMedGPT vs HGT (best baseline, two-sample Welch's t-test) -
# Seeds don't align (NM: 1-5 on TxGNN splits; HGT: 42-46 on own splits)
# → independent samples t-test (Welch's), not paired
print("\nLoading off-label use results...")
RESULTS_DIR = "reproduce_plots/results"
nm_final = pd.read_csv(f"{RESULTS_DIR}/final_results.csv")
hgt      = pd.read_csv(f"{RESULTS_DIR}/HGT_off_label.csv")

nm_off = nm_final[nm_final["task"] == "off-label use"].sort_values("seed")

off_rows = []
for metric, nm_col, hgt_col in [
    ("AUC",        "auc",   "auc"),
    ("AUPR",       "auprc", "auprc"),
    ("Recall@100", "hit",   "recall100"),
]:
    nm_vals   = nm_off[nm_col].values
    comp_vals = hgt[hgt_col].values

    t_stat, p_val = stats.ttest_ind(nm_vals, comp_vals, equal_var=False)
    mean_diff = float(np.mean(nm_vals) - np.mean(comp_vals))

    off_rows.append({
        "metric":      metric,
        "nm_mean":     round(float(np.mean(nm_vals)),  4),
        "nm_std":      round(float(np.std(nm_vals, ddof=1)), 4),
        "hgt_mean":    round(float(np.mean(comp_vals)), 4),
        "hgt_std":     round(float(np.std(comp_vals, ddof=1)), 4),
        "mean_diff":   round(mean_diff, 4),
        "t_stat":      round(float(t_stat), 4),
        "p_value":     round(float(p_val), 4),
        "significant": p_val < 0.05,
    })

off_df = pd.DataFrame(off_rows)

print("\n" + "=" * 100)
print("Off-label use: NetMedGPT vs HGT (best baseline) — two-sample Welch's t-test (5 seeds each)")
print("Note: different splits used → independent samples test, not paired")
print("=" * 100)
print(f"{'Metric':<14} {'NM mean':>9} {'NM std':>8} {'HGT mean':>10} {'HGT std':>9} "
      f"{'Diff':>8} {'p-value':>9} {'Sig?':>6}")
print("-" * 100)
for _, r in off_df.iterrows():
    sig = "*" if r["significant"] else ""
    print(f"{r['metric']:<14} {r['nm_mean']:>9.4f} {r['nm_std']:>8.4f} "
          f"{r['hgt_mean']:>10.4f} {r['hgt_std']:>9.4f} "
          f"{r['mean_diff']:>8.4f} {r['p_value']:>9.4f} {sig:>6}")

off_df.insert(0, "task", "off-label use")
off_df.insert(1, "comparison", "NetMedGPT vs HGT")
out2 = pd.concat([out, off_df], ignore_index=True)
out2.to_csv("data/result/statistical_significance_dti_adr.csv", index=False)
print(f"\nUpdated: data/result/statistical_significance_dti_adr.csv")


# ── Disease areas: one-sample t-test (NM seed 1 vs TxGNN distribution) ───────
print("\nLoading disease-area results...")
nm_all   = pd.read_csv("data/result/all_checkpoints_evaluation.csv")
txgnn_all = pd.read_csv("/home/bbc8731/txgnn_docker/TxGNN/reproduce/result_more_metrics.csv")
txgnn = txgnn_all[txgnn_all["Method"] == "TxGNN"].copy()

DISEASE_AREAS = [
    "adrenal_gland", "anemia", "autoimmune", "cardiovascular",
    "cell_proliferation", "diabetes", "mental_health",
    "metabolic_disorder", "neurodigenerative",
]
EDGE_TYPES = ["indication", "contraindication"]
METRIC_MAP = {
    "auc":        "AUROC",
    "auprc":      "AUPRC",
    "recall@100": "Recall@100",
}

da_rows = []
for area in DISEASE_AREAS:
    nm_sub  = nm_all[(nm_all["inference"] == area) & (nm_all["edge_type"].isin(EDGE_TYPES))]
    txg_sub = txgnn[txgnn["Split"] == area]

    for edge in EDGE_TYPES:
        nm_e  = nm_sub[nm_sub["edge_type"] == edge]
        txg_e = txg_sub[txg_sub["Task"] == edge]

        for nm_metric, txgnn_metric in METRIC_MAP.items():
            nm_val = nm_e[nm_metric].values
            txg_vals = (
                txg_e[txg_e["Metric Name"] == txgnn_metric]
                .sort_values("Seed")["Metric"].values
            )
            if len(nm_val) == 0 or len(txg_vals) < 2:
                continue

            nm_v     = float(nm_val[0])
            txg_mean = float(np.mean(txg_vals))
            txg_std  = float(np.std(txg_vals, ddof=1))
            n        = len(txg_vals)

            # one-sample t-test: is nm_v a plausible draw from TxGNN's distribution?
            t_stat = (nm_v - txg_mean) / (txg_std / np.sqrt(n))
            p_val  = float(2 * stats.t.sf(abs(t_stat), df=n - 1))  # two-sided

            da_rows.append({
                "area":       area,
                "edge_type":  edge,
                "metric":     nm_metric,
                "nm_value":   round(nm_v,     4),
                "txgnn_mean": round(txg_mean, 4),
                "txgnn_std":  round(txg_std,  4),
                "diff":       round(nm_v - txg_mean, 4),
                "t_stat":     round(t_stat,   4),
                "p_value":    round(p_val,    4),
                "significant": p_val < 0.05,
            })

da_df = pd.DataFrame(da_rows)

print("\n" + "=" * 115)
print("DISEASE AREAS: NetMedGPT (seed 1) vs TxGNN — one-sample t-test (df=4)")
print("Note: NM has 1 seed per area; test asks if NM falls outside TxGNN's 5-seed distribution")
print("=" * 115)
print(f"{'Area':<22} {'Edge':<18} {'Metric':<12} {'NM':>8} {'TxGNN mean':>11} "
      f"{'TxGNN std':>10} {'Diff':>8} {'p-value':>9} {'Sig?':>6}")
print("-" * 115)
for _, r in da_df.sort_values(["area", "edge_type", "metric"]).iterrows():
    sig = "*" if r["significant"] else ""
    print(f"{r['area']:<22} {r['edge_type']:<18} {r['metric']:<12} {r['nm_value']:>8.4f} "
          f"{r['txgnn_mean']:>11.4f} {r['txgnn_std']:>10.4f} {r['diff']:>8.4f} "
          f"{r['p_value']:>9.4f} {sig:>6}")

# summary: how many significant
n_sig = da_df["significant"].sum()
n_total = len(da_df)
print(f"\nSignificant comparisons: {n_sig} / {n_total}")
print(f"NetMedGPT better (diff > 0): {(da_df['diff'] > 0).sum()} / {n_total}")

da_df.to_csv("data/result/statistical_significance_disease_areas.csv", index=False)
print(f"Saved to data/result/statistical_significance_disease_areas.csv")
