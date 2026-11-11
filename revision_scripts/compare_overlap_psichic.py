"""
NetMedGPT vs PSICHIC on the drug_protein task (random split) on exactly the same pairs.

Scores:
  NetMedGPT  data/new_analysis/overlap/netmedgpt_random_link_split_drug_protein_seed<N>.pt (export_netmedgpt_scores.py)
  PSICHIC    pred_val.csv / pred_test.csv of psichic_baseline.py (data/new_analysis/psichic/random_link_split_seed<N>/),
             with the column covered = pair PSICHIC can score (drug with SMILES, protein sequence <= 2500 residues)

Both methods are evaluated on all pairs and on the overlapping pairs only (covered), with the same rules: AUROC and
AUPRC on the test pairs; F1 with the threshold that maximises F1 on the val pairs of the same subset.
Recall@100 / Precision@100 only on all pairs (PSICHIC's per-drug ranking is not saved; its value is read from
hit_precision_test.csv of psichic_baseline.py, NetMedGPT's from the export).

Statistics (when results of at least 2 seeds exist): two-sided paired t-test over the seeds, Holm correction over the
metrics within each pair set, bootstrap 95% CI (same functions as compare_overlap_deepadr.py).
Seeds without PSICHIC results are skipped.

Output: data/result/new_analysis/overlap_netmedgpt_vs_psichic_per_seed.csv
        data/result/new_analysis/overlap_netmedgpt_vs_psichic.csv

Example (from the repository root, after export_netmedgpt_scores.py --task drug_protein):
    python compare_overlap_psichic.py
"""

import os
import argparse
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score, average_precision_score

from compare_overlap_deepadr import f1_val_threshold, holm, bootstrap_ci


parser = argparse.ArgumentParser()
parser.add_argument("--seeds", required=False, default="1,2,3,4,5")
parser.add_argument("--scores_dir", required=False, default="data/new_analysis/overlap")
parser.add_argument("--psichic_dir", required=False, default="data/new_analysis/psichic")
parser.add_argument("--result_dir", required=False, default="data/result/new_analysis")

TASK = "drug_protein"
INFERENCE = "random_link_split"


def scores(val_y, val_s, test_y, test_s):
    return {'AUROC': roc_auc_score(test_y, test_s), 'AUPRC': average_precision_score(test_y, test_s),
            'F1': f1_val_threshold(val_y, val_s, test_y, test_s)}


def main():
    args = parser.parse_args()
    rows = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        pdir = os.path.join(args.psichic_dir, f"{INFERENCE}_seed{seed}")
        hp_path = os.path.join(args.result_dir, f"psichic_{INFERENCE}_seed{seed}", "hit_precision_test.csv")
        if not (os.path.exists(os.path.join(pdir, 'pred_test.csv')) and os.path.exists(hp_path)):
            print(f"seed {seed}: no PSICHIC results yet, skipped")
            continue
        nm = torch.load(os.path.join(args.scores_dir, f"netmedgpt_{INFERENCE}_{TASK}_seed{seed}.pt"))
        ps = {n: pd.read_csv(os.path.join(pdir, f'pred_{n}.csv')) for n in ['val', 'test']}
        for n in ['val', 'test']:  # same pairs in the same order
            idx = nm[f'{n}_idx']
            assert (ps[n]['head'].values == idx[0].numpy()).all() and (ps[n]['tail'].values == idx[2].numpy()).all()
            assert (ps[n]['classification_label'].values == nm[f'{n}_label'].numpy()).all()

        hp = pd.read_csv(hp_path, index_col=0)
        recall = {'NetMedGPT': (float(nm['recall'].mean()), float(nm['precision'].mean())),
                  'PSICHIC': (float(hp.loc[TASK, 'hit']), float(hp.loc[TASK, 'precision']))}
        for subset, (mv, mt) in {'all pairs': (slice(None), slice(None)),
                                 'overlap (pairs PSICHIC can score)': (ps['val']['covered'].values, ps['test']['covered'].values)}.items():
            vy, ty = nm['val_label'].numpy()[mv], nm['test_label'].numpy()[mt]
            for method, (vs, ts) in [('NetMedGPT', (nm['val_score'].numpy(), nm['test_score'].numpy())),
                                     ('PSICHIC', (ps['val']['score'].values, ps['test']['score'].values))]:
                row = {'seed': seed, 'pairs': subset, 'method': method, 'n_test_pairs': len(ty),
                       **scores(vy, vs[mv], ty, ts[mt])}
                if subset == 'all pairs':
                    row['Recall@100'], row['Precision@100'] = recall[method]
                rows.append(row)
        print(f"seed {seed}: done")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(args.result_dir, "overlap_netmedgpt_vs_psichic_per_seed.csv"), index=False)
    with pd.option_context('display.width', 200, 'display.float_format', '{:.4f}'.format):
        print(per_seed.to_string(index=False))

    if per_seed['seed'].nunique() < 2:
        print("statistics need at least 2 seeds")
        return
    summary = []
    for subset, g in per_seed.groupby('pairs', sort=False):
        part = []
        for metric in [m for m in ['AUROC', 'AUPRC', 'F1', 'Recall@100', 'Precision@100'] if g[m].notna().all()]:
            a = g[g['method'] == 'NetMedGPT'].set_index('seed')[metric].sort_index()
            b = g[g['method'] == 'PSICHIC'].set_index('seed')[metric].sort_index()
            d = (a - b).values
            lo, hi = bootstrap_ci(d)
            part.append({'pairs': subset, 'metric': metric, 'NetMedGPT_mean': a.mean(), 'NetMedGPT_std': a.std(),
                         'PSICHIC_mean': b.mean(), 'PSICHIC_std': b.std(), 'mean_diff': d.mean(),
                         'ci_low': lo, 'ci_high': hi, 'p_value': ttest_rel(a.values, b.values).pvalue, 'n_seeds': len(d)})
        for r, p in zip(part, holm([r['p_value'] for r in part])):
            r['p_holm'], r['significant'] = p, p < 0.05
        summary += part
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.result_dir, "overlap_netmedgpt_vs_psichic.csv"), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.float_format', '{:.4f}'.format):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
