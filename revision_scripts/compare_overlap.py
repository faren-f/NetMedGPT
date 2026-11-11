"""
NetMedGPT vs a baseline on exactly the same pairs, from saved per-pair scores:
  NetMedGPT  data/new_analysis/overlap/netmedgpt_<split>_<task>_seed<N>.pt  (export_netmedgpt_scores.py)
  baseline   data/new_analysis/overlap/<baseline>_<split>_<task>_seed<N>.pt  (e.g. galeano_baseline.py), with
             the masks covered_val / covered_test / covered_heads of the pairs and drugs the baseline can score

Both methods are evaluated on all pairs and on the overlapping pairs only (pairs the baseline can score), with the
same rules: AUROC and AUPRC on the test pairs; F1 with the threshold that maximises F1 on the val pairs of the same
subset; Recall@100 / Precision@100 per test drug over all candidate tails, averaged over the drugs of the subset.

Statistics: two-sided paired t-test over the seeds (NetMedGPT - baseline), Holm correction over the metrics within
each pair set, bootstrap 95% CI of the mean difference (same as compare_overlap_deepadr.py).

Output: data/result/new_analysis/overlap_netmedgpt_vs_<baseline>_per_seed.csv
        data/result/new_analysis/overlap_netmedgpt_vs_<baseline>.csv

Example (from the repository root):
    python compare_overlap.py --baseline galeano --task drug_effect
"""

import os
import argparse
import pandas as pd
import torch
from scipy.stats import ttest_rel

from compare_overlap_deepadr import metrics, holm, bootstrap_ci, METRICS


parser = argparse.ArgumentParser()
parser.add_argument("--baseline", required=True, help="method name of the saved scores, e.g. galeano")
parser.add_argument("--task", required=False, default="drug_effect")
parser.add_argument("--inference", required=False, default="random_link_split")
parser.add_argument("--seeds", required=False, default="1,2,3,4,5")
parser.add_argument("--scores_dir", required=False, default="data/new_analysis/overlap")
parser.add_argument("--out_dir", required=False, default="data/result/new_analysis")


def main():
    args = parser.parse_args()
    name = {'galeano': 'Galeano'}.get(args.baseline, args.baseline)
    rows = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        nm = torch.load(os.path.join(args.scores_dir, f"netmedgpt_{args.inference}_{args.task}_seed{seed}.pt"))
        bl = torch.load(os.path.join(args.scores_dir, f"{args.baseline}_{args.inference}_{args.task}_seed{seed}.pt"))
        for key in ['val_idx', 'val_label', 'test_idx', 'test_label', 'heads']:  # same pairs in the same order
            assert torch.equal(nm[key], bl[key]), key

        subsets = {'all pairs': (torch.ones(len(nm['val_label']), dtype=torch.bool),
                                 torch.ones(len(nm['test_label']), dtype=torch.bool),
                                 torch.ones(len(nm['heads']), dtype=torch.bool)),
                   f'overlap (pairs {name} can score)': (bl['covered_val'], bl['covered_test'], bl['covered_heads'])}
        for subset, (mv, mt, mh) in subsets.items():
            for method, s in [('NetMedGPT', nm), (name, bl)]:
                res = metrics(nm['val_label'][mv].numpy(), s['val_score'][mv].numpy(),
                              nm['test_label'][mt].numpy(), s['test_score'][mt].numpy(),
                              s['recall'][mh], s['precision'][mh])
                rows.append({'seed': seed, 'pairs': subset, 'method': method, 'n_test_pairs': int(mt.sum()),
                             'n_test_drugs': int(mh.sum()), **res})

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(args.out_dir, f"overlap_netmedgpt_vs_{args.baseline}_per_seed.csv"), index=False)

    summary = []
    for subset, g in per_seed.groupby('pairs', sort=False):
        part = []
        for metric in METRICS:
            a = g[g['method'] == 'NetMedGPT'].set_index('seed')[metric].sort_index()
            b = g[g['method'] == name].set_index('seed')[metric].sort_index()
            d = (a - b).values
            lo, hi = bootstrap_ci(d)
            part.append({'pairs': subset, 'metric': metric,
                         'NetMedGPT_mean': a.mean(), 'NetMedGPT_std': a.std(),
                         f'{name}_mean': b.mean(), f'{name}_std': b.std(),
                         'mean_diff': d.mean(), 'ci_low': lo, 'ci_high': hi,
                         'p_value': ttest_rel(a.values, b.values).pvalue, 'n_seeds': len(d),
                         'n_test_pairs': int(g['n_test_pairs'].iloc[0]), 'n_test_drugs': int(g['n_test_drugs'].iloc[0])})
        for r, p in zip(part, holm([r['p_value'] for r in part])):
            r['p_holm'] = p
            r['significant'] = p < 0.05
        summary += part
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.out_dir, f"overlap_netmedgpt_vs_{args.baseline}.csv"), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.float_format', '{:.4f}'.format):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
