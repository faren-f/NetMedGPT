"""
Galeano et al. (Nature Communications 2020, "Predicting the frequencies of drug side effects") on the drug_effect task
of the NetMedGPT random split.

The model is the non-negative matrix decomposition of the drug x side effect matrix, R ~ W H (W, H >= 0), with
weight 1 on the observed entries and weight alpha on the zeros. decomposition() is a line-by-line port of
DecompositionAlgorithm.m of the official code (https://github.com/paccanarolab/Side-effect-Frequencies, commit 34c1b3a,
MATLAB), with the same initialisation, multiplicative updates, normalisation of H and stopping rule
(tolx 1e-3, maxiter 2000). Parameters as in the paper: k = 10 latent features, alpha = 0.05.

Data:
  R = drugs x effects matrix of the drug_effect edges of the TxGNN train split (1 = edge, 0 = no edge; PrimeKG has
      no frequency classes, so the observed entries are 1, as for presence/absence in the paper).
      Rows / columns are the drugs / effects with at least one training edge, as in the paper's matrix.
  Scores = entries of W H. Pairs whose drug or effect is not in the training matrix cannot be scored and get 0.
  Val/test pairs: the NetMedGPT drug_effect val/test pairs from the corpus cache.

Evaluation: same functions and files as NetMedGPT and the other baselines (AUROC, AUPRC, F1 with the val threshold,
Recall@100 / Precision@100 over all effect/phenotype nodes). Also saved:
  LP_test_covered.csv  metrics on the test pairs whose drug and effect are in the training matrix
  data/new_analysis/overlap/galeano_<split>_drug_effect_seed<N>.pt  per-pair scores and per-drug Recall@100
      (same format as export_netmedgpt_scores.py, for compare_overlap.py)

Method name in the results: galeano (only the drug_effect task, random_link_split)

Example (from the repository root, in the netmedgpt environment):
    python galeano_baseline.py --seed 1 --gpu 0
"""

import os
import argparse
import numpy as np
import torch

from common import load_split, build_corpus, get_device, PairScorer, score_pairs, training_pairs, f1_per_edge_type
from netmedgpt.utilities import link_pred_val
from baseline_utils import finish, work_dir, relation_tokens


####### arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID")
parser.add_argument("--seed", required=False, default=1, type=int, help="Random seed / split number")
parser.add_argument("--k", required=False, default=10, type=int, help="latent features (paper: 10)")
parser.add_argument("--alpha", required=False, default=0.05, type=float, help="confidence on the zeros (paper: 0.05)")
parser.add_argument("--tolx", required=False, default=1e-3, type=float, help="termination tolerance (official code: 1e-3)")
parser.add_argument("--maxiter", required=False, default=2000, type=int, help="maximum iterations (official code: 2000)")
parser.add_argument("--recall_k", required=False, default=100, type=int)
parser.add_argument("--overlap_dir", required=False, default="data/new_analysis/overlap")

INFERENCE = "random_link_split"
TASK = "drug_effect"


#######################################################################################################################
# Decomposition algorithm (port of DecompositionAlgorithm.m)

# MATLAB eps(x): distance from |x| to the next larger double
def matlab_eps(x):
    a = x.abs()
    return torch.nextafter(a, torch.full_like(a, float('inf'))) - a


# R: drugs x effects (float64). W0, H0: optional initial factors (otherwise rand * sqrt(0.01), as in the official
# code). Returns W, H (H as returned by the official code at termination) and the cost per iteration.
def decomposition(R, k, alpha, tolx=1e-3, maxiter=2000, W0=None, H0=None, generator=None, verbose=False):
    variance = 0.01
    ndrugs, nses = R.shape
    if W0 is None:
        W0 = torch.rand(ndrugs, k, dtype=R.dtype, generator=generator).to(R.device) * np.sqrt(variance)
    if H0 is None:
        H0 = torch.rand(k, nses, dtype=R.dtype, generator=generator).to(R.device) * np.sqrt(variance)
    H0 = H0 / torch.sqrt((H0 ** 2).sum(1, keepdim=True))
    sqrteps = np.sqrt(np.finfo(np.float64).eps)
    J = []

    CT = (R > 0).to(R.dtype)   # observed associations
    UN = (R == 0).to(R.dtype)  # unobserved associations
    CTR = CT * R

    for it in range(1, maxiter + 1):
        numer = CTR @ H0.T
        WH = W0 @ H0
        W = torch.clamp_min(W0 * (numer / ((CT * WH + alpha * UN * WH) @ H0.T + matlab_eps(numer))), 0)

        numer = W.T @ CTR
        WH = W @ H0
        H = torch.clamp_min(H0 * (numer / (W.T @ (CT * WH + alpha * UN * WH) + matlab_eps(numer))), 0)

        E = R - W @ H
        J.append(float(0.5 * ((CT * E) ** 2).sum() + 0.5 * alpha * ((UN * E) ** 2).sum()))

        dw = ((W - W0).abs() / (sqrteps + W0.abs().max())).max()
        dh = ((H - H0).abs() / (sqrteps + H0.abs().max())).max()
        delta = float(max(dw, dh))
        if verbose and it % 100 == 0:
            print(f"  iter {it} cost {J[-1]:.6e} delta {delta:.3e}")
        if it > 1 and delta <= tolx:
            break

        W0 = W
        H0 = H / torch.sqrt((H ** 2).sum(1, keepdim=True))
    return W, H, J


#######################################################################################################################

def task_only(split, data):
    r = split['queried_edge_types'].index(TASK)
    split = dict(split, queried_edge_types=[TASK], queried_node_types=[split['queried_node_types'][r]],
                 edge_flags=[split['edge_flags'][r]])
    z = relation_tokens(split)[TASK]
    out = {}
    for n in ['val', 'test']:
        m = data[f'{n}_data_edge_label_index_with_relation'][1] == z
        out[f'{n}_data_edge_label_index_with_relation'] = data[f'{n}_data_edge_label_index_with_relation'][:, m]
        out[f'{n}_data_edge_label'] = data[f'{n}_data_edge_label'][m]
        out[f'edge_type_flag_{n}'] = data[f'edge_type_flag_{n}'][m]
    return split, out, z


# score of every (drug, effect) node pair from W H; 0 for drugs / effects outside the training matrix
def score_lookup(Rhat, drugs, effects, vocab):
    row = torch.full((vocab,), -1, dtype=torch.long)
    col = torch.full((vocab,), -1, dtype=torch.long)
    row[drugs] = torch.arange(len(drugs))
    col[effects] = torch.arange(len(effects))

    def pair_score_fn(heads, tails, rel=None):
        h, t = row[heads.cpu()], col[tails.cpu()]
        out = torch.zeros(len(h), dtype=torch.float32)
        ok = (h >= 0) & (t >= 0)
        out[ok] = Rhat[h[ok], t[ok]]
        return out

    return pair_score_fn, row, col


# Recall@k / Precision@k per test drug over all candidate effects (as node_level_eval, without averaging)
def per_head_recall(pair_score_fn, idx, label, cands, k):
    pos = idx[:, label == 1]
    heads = pos[0].unique()
    recall, precision = [], []
    for h in heads.tolist():
        s = pair_score_fn(torch.full((len(cands),), h), cands)
        top = cands[torch.topk(s, k=k).indices]
        true = pos[2, pos[0] == h]
        hit = torch.isin(true, top).sum().item()
        recall.append(hit / len(true))
        precision.append(hit / k)
    return heads, torch.tensor(recall), torch.tensor(precision)


def main():
    args = parser.parse_args()
    device = get_device(args.gpu)
    method = "galeano"

    # 1. same split and same val/test pairs as NetMedGPT, drug_effect only
    split = load_split(INFERENCE, args.seed)
    split, data, z = task_only(split, build_corpus(split))
    out_dir = work_dir(split, method)

    # 2. training matrix: drugs x effects with at least one training drug_effect edge
    h, t, y = training_pairs(split, TASK)
    h, t = h[y == 1], t[y == 1]
    drugs, effects = h.unique(), t.unique()
    vocab = int(split['nodes']['node_index'].max()) + 1
    _, row, col = score_lookup(None, drugs, effects, vocab)
    R = torch.zeros(len(drugs), len(effects), dtype=torch.float64)
    R[row[h], col[t]] = 1.0
    print(f"training matrix: {len(drugs)} drugs x {len(effects)} effects, {int(R.sum())} edges")

    # 3. decomposition (official algorithm and parameters)
    g = torch.Generator().manual_seed(args.seed)
    W, H, J = decomposition(R.to(device), args.k, args.alpha, args.tolx, args.maxiter, generator=g, verbose=True)
    print(f"decomposition: {len(J)} iterations, final cost {J[-1]:.6e}")
    Rhat = (W @ H).float().cpu()
    torch.save({'W': W.cpu(), 'H': H.cpu(), 'drugs': drugs, 'effects': effects, 'cost': J},
               os.path.join(out_dir, "factors.pt"))

    # 4. scoring of the val/test pairs and of all candidates (Recall@100), same metrics and files as NetMedGPT
    pair_score_fn, row, col = score_lookup(Rhat, drugs, effects, vocab)
    scorer = PairScorer(pair_score_fn, split['vocab_size'], split['all_ids_node2'], device)
    pred_val = score_pairs(data['val_data_edge_label_index_with_relation'], pair_score_fn)
    pred_test = score_pairs(data['test_data_edge_label_index_with_relation'], pair_score_fn)
    finish(split, data, pred_val, pred_test, scorer, device, method)

    # 5. pairs the model can score (drug and effect in the training matrix) + per-pair scores for compare_overlap.py
    cov = {n: (row[data[f'{n}_data_edge_label_index_with_relation'][0]] >= 0)
              & (col[data[f'{n}_data_edge_label_index_with_relation'][2]] >= 0) for n in ['val', 'test']}
    idx, lab = data['test_data_edge_label_index_with_relation'], data['test_data_edge_label']
    heads, recall, precision = per_head_recall(pair_score_fn, idx, lab, split['all_ids_node2'][z], args.recall_k)
    os.makedirs(args.overlap_dir, exist_ok=True)
    torch.save({'task': TASK, 'seed': args.seed, 'k': args.recall_k,
                'val_idx': data['val_data_edge_label_index_with_relation'], 'val_label': data['val_data_edge_label'],
                'val_score': pred_val, 'test_idx': idx, 'test_label': lab, 'test_score': pred_test,
                'heads': heads, 'recall': recall, 'precision': precision,
                'covered_val': cov['val'], 'covered_test': cov['test'], 'covered_heads': row[heads] >= 0},
               os.path.join(args.overlap_dir, f"{method}_{INFERENCE}_{TASK}_seed{args.seed}.pt"))

    m, mv = cov['test'], cov['val']
    _, thr = f1_per_edge_type(pred_val[mv], data['val_data_edge_label'][mv], data['edge_type_flag_val'][mv], split['edge_flags'])
    covered = link_pred_val(pred_test[m], lab[m], data['edge_type_flag_test'][m], split['all_param'], edge_type=split['edge_flags'])
    covered['f1'], _ = f1_per_edge_type(pred_test[m], lab[m], data['edge_type_flag_test'][m], split['edge_flags'], thresholds=thr)
    covered.index = [TASK]
    covered['n_pairs'], covered['n_pairs_all'], covered['seed'] = int(m.sum()), len(m), args.seed
    result_dir = os.path.join(split['data_dir'], 'result', 'new_analysis', f"{method}_{INFERENCE}_seed{args.seed}")
    covered.to_csv(os.path.join(result_dir, 'LP_test_covered.csv'))
    print("test pairs whose drug and effect are in the training matrix:")
    print(covered)


if __name__ == "__main__":
    main()
