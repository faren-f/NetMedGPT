"""
NetMedGPT vs DeepADR on the drug_effect task (random split, seeds 1-5), on exactly the same pairs.

Both methods are evaluated
  - on all test pairs, and
  - on the overlapping pairs only: pairs whose drug has a SMILES (the pairs DeepADR scores with all its inputs;
    drugs without SMILES get a zero MoLFormer vector in deepadr_baseline.py).
with the same rules for both methods:
  AUROC, AUPRC                 on the test pairs of the subset
  F1                           threshold that maximises F1 on the val pairs of the same subset, applied to test
  Recall@100, Precision@100    per test drug over all effect/phenotype nodes, averaged over the drugs of the subset

Scores:
  NetMedGPT  per-pair scores and per-drug Recall@100 from export_netmedgpt_scores.py (saved checkpoints)
  DeepADR    the best model of deepadr_baseline.py (data/new_analysis/deepadr/<split>_seed<N>/), scored again here

Statistics: paired t-test over the 5 seeds (NetMedGPT - DeepADR), Holm correction over the metrics within each
pair set, bootstrap 95% CI of the mean difference.

Output: data/result/new_analysis/overlap_netmedgpt_vs_deepadr_per_seed.csv
        data/result/new_analysis/overlap_netmedgpt_vs_deepadr.csv

Example (from the repository root, in the deepadr environment, after export_netmedgpt_scores.py --task drug_effect):
    python compare_overlap_deepadr.py --gpu 1 --deepadr_dir /home/bbc8731/DeepADR
"""

import os
import types
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

from common import load_split, build_corpus, get_device
from baseline_utils import work_dir, in_dir
from deepadr_baseline import load_deepadr, task_only, build_features, pair_probs, INFERENCE, TASK


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID")
parser.add_argument("--deepadr_dir", required=True, help="path to the cloned DeepADR repository")
parser.add_argument("--inputs_dir", required=False, default="data/new_analysis/adr_inputs")
parser.add_argument("--scores_dir", required=False, default="data/new_analysis/overlap", help="output of export_netmedgpt_scores.py")
parser.add_argument("--seeds", required=False, default="1,2,3,4,5")
parser.add_argument("--kan_hidden_dim", required=False, default=16, type=int, help="as used in training (DeepADR default 16)")
parser.add_argument("--eval_batch_size", required=False, default=16384, type=int)
parser.add_argument("--out_dir", required=False, default="data/result/new_analysis")

METRICS = ['AUROC', 'AUPRC', 'F1', 'Recall@100', 'Precision@100']


# DeepADR best model of deepadr_baseline.py for one seed
def load_deepadr_model(args, deepadr_train, split, feats):
    shapes = types.SimpleNamespace(drug_molformer=feats['mol'], drug_target=feats['target'], adr_features=feats['adr'])
    out_dir = work_dir(split, "deepadr")
    with in_dir(out_dir):
        model = deepadr_train.build_model(shapes, args)
    model.kan_layer.speed()
    model.load_state_dict(torch.load(os.path.join(out_dir, "deepadr_classification_best.pt"), map_location=args.device))
    model.eval()
    return model


# Recall@k / Precision@k per test drug: all candidate effects scored, top k compared with the test positives
def deepadr_per_head_recall(model, feats, idx, label, heads, cands, args, k):
    pos = idx[:, label == 1]
    recall, precision = [], []
    for h in heads.tolist():
        s = pair_probs(model, feats, torch.full((len(cands),), h), cands, args.device, args.eval_batch_size)
        top = cands[torch.topk(s, k=k).indices]
        true = pos[2, pos[0] == h]
        hit = torch.isin(true, top).sum().item()
        recall.append(hit / len(true))
        precision.append(hit / k)
    return torch.tensor(recall), torch.tensor(precision)


def f1_val_threshold(val_y, val_s, test_y, test_s):
    prec, rec, thr = precision_recall_curve(val_y, val_s)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    t = thr[np.argmax(f1[:-1])] if len(thr) else 0.5
    return f1_score(test_y, (test_s >= t).astype(int), zero_division=0)


def metrics(val_y, val_s, test_y, test_s, recall, precision):
    return {'AUROC': roc_auc_score(test_y, test_s), 'AUPRC': average_precision_score(test_y, test_s),
            'F1': f1_val_threshold(val_y, val_s, test_y, test_s),
            'Recall@100': float(recall.mean()), 'Precision@100': float(precision.mean())}


def holm(p):
    p = np.asarray(p, dtype=float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def bootstrap_ci(diffs, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    args = parser.parse_args()
    args.device = str(get_device(args.gpu))
    deepadr_train = load_deepadr(args.deepadr_dir)

    rows = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        nm = torch.load(os.path.join(args.scores_dir, f"netmedgpt_{INFERENCE}_{TASK}_seed{seed}.pt"))
        split = load_split(INFERENCE, seed)
        split, data, z = task_only(split, build_corpus(split))
        feats = build_features(split, args.inputs_dir)
        for n in ['val', 'test']:  # same pairs in the same order
            assert torch.equal(nm[f'{n}_idx'], data[f'{n}_data_edge_label_index_with_relation'])
            assert torch.equal(nm[f'{n}_label'], data[f'{n}_data_edge_label'])

        model = load_deepadr_model(args, deepadr_train, split, feats)
        da = {}
        for n in ['val', 'test']:
            idx = nm[f'{n}_idx']
            da[f'{n}_score'] = pair_probs(model, feats, idx[0], idx[2], args.device, args.eval_batch_size)
        da['recall'], da['precision'] = deepadr_per_head_recall(
            model, feats, nm['test_idx'], nm['test_label'], nm['heads'], split['all_ids_node2'][z], args, nm['k'])

        has_smiles = feats['has_smiles']
        subsets = {'all pairs': (torch.ones(len(nm['val_label']), dtype=torch.bool),
                                 torch.ones(len(nm['test_label']), dtype=torch.bool),
                                 torch.ones(len(nm['heads']), dtype=torch.bool)),
                   'overlap (drug has SMILES)': (has_smiles[nm['val_idx'][0]], has_smiles[nm['test_idx'][0]],
                                                 has_smiles[nm['heads']])}
        for subset, (mv, mt, mh) in subsets.items():
            for method, s in [('NetMedGPT', nm), ('DeepADR', da)]:
                res = metrics(nm['val_label'][mv].numpy(), s['val_score'][mv].numpy(),
                              nm['test_label'][mt].numpy(), s['test_score'][mt].numpy(),
                              s['recall'][mh], s['precision'][mh])
                rows.append({'seed': seed, 'pairs': subset, 'method': method, 'n_test_pairs': int(mt.sum()),
                             'n_test_drugs': int(mh.sum()), **res})
        print(f"seed {seed} done: test pairs {len(nm['test_label'])}, overlap {int(subsets['overlap (drug has SMILES)'][1].sum())}")
        del model
        torch.cuda.empty_cache()

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(args.out_dir, "overlap_netmedgpt_vs_deepadr_per_seed.csv"), index=False)

    summary = []
    for subset, g in per_seed.groupby('pairs', sort=False):
        part = []
        for metric in METRICS:
            a = g[g['method'] == 'NetMedGPT'].set_index('seed')[metric].sort_index()
            b = g[g['method'] == 'DeepADR'].set_index('seed')[metric].sort_index()
            d = (a - b).values
            lo, hi = bootstrap_ci(d)
            part.append({'pairs': subset, 'metric': metric,
                         'NetMedGPT_mean': a.mean(), 'NetMedGPT_std': a.std(),
                         'DeepADR_mean': b.mean(), 'DeepADR_std': b.std(),
                         'mean_diff': d.mean(), 'ci_low': lo, 'ci_high': hi,
                         'p_value': ttest_rel(a.values, b.values).pvalue, 'n_seeds': len(d),
                         'n_test_pairs': int(g['n_test_pairs'].iloc[0]), 'n_test_drugs': int(g['n_test_drugs'].iloc[0])})
        for r, p in zip(part, holm([r['p_value'] for r in part])):
            r['p_holm'] = p
            r['significant'] = p < 0.05
        summary += part
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.out_dir, "overlap_netmedgpt_vs_deepadr.csv"), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.float_format', '{:.4f}'.format):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
