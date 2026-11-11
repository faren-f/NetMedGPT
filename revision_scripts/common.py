"""
Shared utilities for the walk-embedding baselines
(node2vec, node2vec + relation tokens, DREAMwalk).

Every baseline is evaluated on exactly the same data as NetMedGPT:
same splits, same Node2Vec walk corpus, same val/test positive and negative pairs,
and the same metrics (AUC, AUPRC, F1 with the val-set threshold, Recall@100, Precision@100).

"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_curve

from netmedgpt.utilities import link_pred_val, node_level_eval, sim
from netmedgpt.prepare_txgnn_splits import prepare_txgnn_data
from netmedgpt.sentence_generation import sentence_generation, negative_samples
from config import set_seed, print_reproducibility_settings


#######################################################################################################################
# Parameters and device
def load_params(path="data/parameters.json"):
    with open(path, 'r') as file:
        all_param = json.load(file)
    return all_param


def get_device(gpu=0):
    return torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')


#######################################################################################################################
# Split configuration and data loading

def get_split_config(inference, seed):
    if inference == "random_link_split":
        cfg = dict(
            queried_edge_types=['indication', 'contraindication', 'off-label use', 'drug_protein', 'drug_effect'],
            queried_node_types=['disease|drug', 'disease|drug', 'disease|drug', 'drug|gene/protein', 'drug|effect/phenotype'],
            edge_flags=[1, 2, 3, 4, 5],
            split_dir=f"data/splits/random_{seed}",
            read_kwargs={},
            dropna=False,
        )
    elif inference == "zero_shot_split":
        cfg = dict(
            queried_edge_types=['indication', 'contraindication'],
            queried_node_types=['disease|drug', 'disease|drug'],
            edge_flags=[1, 2],
            split_dir=f"data/splits/complex_disease_{seed}",
            read_kwargs={},
            dropna=False,
        )
    elif inference == "rare_disease":
        cfg = dict(
            queried_edge_types=['indication', 'contraindication', 'off-label use'],
            queried_node_types=['disease|drug', 'disease|drug', 'disease|drug'],
            edge_flags=[1, 2, 3],
            split_dir=f"data/splits/rare_disease_{seed}",
            read_kwargs={'low_memory': False},
            dropna=False,
        )
    else:
        # disease-area split (adrenal_gland, anemia, ...)
        cfg = dict(
            queried_edge_types=['indication', 'contraindication'],
            queried_node_types=['disease|drug', 'disease|drug'],
            edge_flags=[1, 2],
            split_dir=f"data/splits/{inference}_{seed}",
            read_kwargs={},
            dropna=True,
        )
    return cfg


def _read_split_csv(path, cfg):
    df = pd.read_csv(path, **cfg['read_kwargs'])
    return df.dropna() if cfg['dropna'] else df


# Loads the KG and the split, and builds the same lookup objects as train.py
# (type2order, relation_node_types, all_ids_node2, mask_token, vocab_size, seq_len).
# Changes:
#   - KG node features (embeddings.pt) are loaded only if requested, since the walk baselines do not use them
def load_split(inference, seed, param_path="data/parameters.json", load_features=False):
    set_seed(seed)
    print_reproducibility_settings(seed)

    all_param = load_params(param_path)
    data_dir = all_param['files']['data_dir']

    nodes = pd.read_csv(os.path.join(data_dir, 'KG/nodes.csv'), sep=',')
    edge = pd.read_csv(os.path.join(data_dir, "KG/edges.csv"))
    feat = torch.load(os.path.join(data_dir, "KG/embeddings.pt")) if load_features else None

    cfg = get_split_config(inference, seed)
    train_txgnn = _read_split_csv(os.path.join(cfg['split_dir'], "train.csv"), cfg)
    val_txgnn = _read_split_csv(os.path.join(cfg['split_dir'], "valid.csv"), cfg)
    test_txgnn = _read_split_csv(os.path.join(cfg['split_dir'], "test.csv"), cfg)

    type2order = dict(zip(cfg['queried_edge_types'], cfg['queried_node_types']))
    train_txgnn, val_txgnn, test_txgnn = prepare_txgnn_data(nodes, train_txgnn, val_txgnn, test_txgnn, type2order)

    with open(f"{data_dir}/KG/relation_node_types.pkl", "rb") as f:
        relation_node_types = pickle.load(f)

    # candidate tail nodes per relation token (used for Recall@100 / Precision@100)
    all_ids = {
        'drug': torch.tensor(
            nodes.loc[nodes['node_type'] == 'drug', 'node_index'].unique(), dtype=torch.int64
        ),
        'gene': torch.tensor(
            nodes.loc[nodes['node_type'] == 'gene/protein', 'node_index'].unique(), dtype=torch.int64
        ),
        'phenotype': torch.tensor(
            nodes.loc[nodes['node_type'] == 'effect/phenotype', 'node_index'].unique(), dtype=torch.int64
        )
    }

    node_type_to_z = edge.groupby('node_types')['z_index'].unique().to_dict()

    all_ids_node2 = {}
    for z in node_type_to_z.get('disease|drug', []):
        all_ids_node2[z] = all_ids['drug']

    for z in node_type_to_z.get('drug|gene/protein', []):
        all_ids_node2[z] = all_ids['gene']

    for z in node_type_to_z.get('drug|effect/phenotype', []):
        all_ids_node2[z] = all_ids['phenotype']

    mask_token = edge['z_index'].max() + 1
    vocab_size = mask_token + 1
    seq_len = (all_param['node2vec']['walk_length'] * 2) - 1

    return dict(
        inference=inference,
        seed=seed,
        all_param=all_param,
        data_dir=data_dir,
        nodes=nodes,
        edge=edge,
        feat=feat,
        train_txgnn=train_txgnn,
        val_txgnn=val_txgnn,
        test_txgnn=test_txgnn,
        relation_node_types=relation_node_types,
        queried_edge_types=cfg['queried_edge_types'],
        queried_node_types=cfg['queried_node_types'],
        edge_flags=cfg['edge_flags'],
        all_ids_node2=all_ids_node2,
        mask_token=mask_token,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )


#######################################################################################################################
# Walk corpus (identical to the one NetMedGPT is trained on)

# same sentence_generation call and arguments as train.py
def _run_sentence_generation(split):
    return sentence_generation(
        split['edge'], split['nodes'], split['queried_node_types'], split['queried_edge_types'],
        split['train_txgnn'], split['val_txgnn'], split['test_txgnn'],
        split['relation_node_types'], split['all_param'], split['seed'])


# builds the corpus once per split/seed, so every baseline reads the
# same walks and the same val/test pairs. Returned dict has the keys of sentence_generation():
#   node2vec_walks                              [N_walks, seq_len] = [n0, r01, n1, r12, n2, ...]
#   val/test_data_edge_label_index_with_relation [3, N_pairs]      = [head, relation, tail]
#   val/test_data_edge_label                    1 = positive, 0 = negative
#   edge_type_flag_val/test                     1..len(queried_edge_types)
def build_corpus(split, cache_dir="data/new_analysis/corpus", use_cache=True):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split['inference']}_seed{split['seed']}.pt")

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached corpus: {cache_path}")
        return torch.load(cache_path)

    data = _run_sentence_generation(split)
    torch.save(data, cache_path)
    print(f"Corpus saved: {cache_path}")
    return data


# checks whether sentence_generation returns the same corpus when called twice with the same seed.
# If True, the regenerated corpus is the one the existing NetMedGPT checkpoints were trained on.
def corpus_is_reproducible(split):
    set_seed(split['seed'])
    a = _run_sentence_generation(split)
    set_seed(split['seed'])
    b = _run_sentence_generation(split)
    same = {k: torch.equal(a[k], b[k]) for k in a}
    for k, v in same.items():
        print(f"{k:<45} identical: {v}")
    return all(same.values())


# walk corpus -> token sentences for skip-gram models (word2vec / DREAMwalk).
#   keep_relations=False : [n0, n1, n2, n3, n4]                   (classic node2vec)
#   keep_relations=True  : [n0, r01, n1, r12, n2, r23, n3, r34, n4] (same sentences NetMedGPT sees)
def walks_to_sentences(walks, keep_relations=False):
    w = walks if keep_relations else walks[:, 0::2]
    return [[str(t) for t in row] for row in w.tolist()]


#######################################################################################################################
# Embeddings -> scores

# token -> vector lookup (e.g. gensim KeyedVectors or a plain dict, keys are token ids as strings)
# dense matrix indexed by token id. Tokens that never occur in the walks get a zero row and are flagged.
def embedding_matrix(token2vec, vocab_size, dim):
    E = torch.zeros(vocab_size, dim, dtype=torch.float32)
    missing = torch.ones(vocab_size, dtype=torch.bool)
    for t in range(vocab_size):
        key = str(t)
        if key in token2vec:
            E[t] = torch.as_tensor(np.asarray(token2vec[key]), dtype=torch.float32)
            missing[t] = False
    return E, missing


# how many val/test head and tail nodes have no embedding (relevant mainly for the zero-shot split)
def coverage_report(missing, data):
    for name in ['val', 'test']:
        idx = data[f'{name}_data_edge_label_index_with_relation']
        heads, tails = idx[0].unique(), idx[2].unique()
        print(f"{name}: heads without embedding {missing[heads].sum().item()}/{len(heads)} | "
              f"tails without embedding {missing[tails].sum().item()}/{len(tails)}")


# adapter so that the existing sim() and node_level_eval() (netmedgpt/utilities.py) can evaluate
# an embedding baseline without any change. Both functions call model(paths) with
# paths = [head, relation, mask, ...] and read out[:, 2, :] as the score of every token as the tail.
# Here out[:, 2, t] = cosine(E[head], E[t]). The relation token is ignored, because plain node2vec
# embeddings are relation-agnostic. The output is an expanded view, so positions 0-2 share memory.
class EmbeddingScorer(nn.Module):
    def __init__(self, E, device):
        super().__init__()
        self.E = F.normalize(E.float(), dim=1).to(device)

    def forward(self, paths):
        scores = self.E[paths[:, 0]] @ self.E.T              # (B, vocab_size)
        return scores.unsqueeze(1).expand(-1, 3, -1)         # (B, 3, vocab_size), only [:, 2, :] is read


# same adapter idea for baselines that score (head, tail) pairs with a trained classifier
# (e.g. node2vec + logistic regression, DREAMwalk + XGBoost).
# pair_score_fn(heads, tails, relation_token) -> 1D tensor/array of scores (higher = more likely).
# Only the candidate tails of each relation (all_ids_node2, built as in train.py) are scored; every other
# token gets -inf, so node_level_eval ranks over the same candidates as for NetMedGPT.
# Use it for node_level_eval; for AUC/AUPRC/F1 use score_pairs() below instead of sim().
class PairScorer(nn.Module):
    def __init__(self, pair_score_fn, vocab_size, all_ids_node2, device, chunk_size=200000):
        super().__init__()
        self.pair_score_fn = pair_score_fn
        self.vocab_size = int(vocab_size)
        self.all_ids_node2 = all_ids_node2
        self.device = device
        self.chunk_size = chunk_size

    def forward(self, paths):
        paths = paths.to(self.device)
        scores = torch.full((paths.shape[0], self.vocab_size), float('-inf'), device=self.device)

        for rel in paths[:, 1].unique():
            rows = (paths[:, 1] == rel).nonzero(as_tuple=True)[0]
            cand = self.all_ids_node2[rel.item()].to(self.device)
            heads = paths[rows, 0].repeat_interleave(len(cand))
            tails = cand.repeat(len(rows))

            s = []
            for i in range(0, len(heads), self.chunk_size):
                s_i = self.pair_score_fn(heads[i:i + self.chunk_size], tails[i:i + self.chunk_size], rel.item())
                s.append(torch.as_tensor(s_i, dtype=torch.float32, device=self.device))
            s = torch.cat(s).view(len(rows), len(cand))

            scores[rows.unsqueeze(1), cand.unsqueeze(0)] = s

        return scores.unsqueeze(1).expand(-1, 3, -1)


# same sim() call as for NetMedGPT, with a baseline scorer as the model
def predict_with_sim(scorer, data, split, device, batch_size=None):
    bs = batch_size or split['all_param']['best_hyperparam']['batch_size']
    pred_val = sim(data['val_data_edge_label_index_with_relation'], scorer, batch_size=bs, method='prob',
                   device=device, mask_token=split['mask_token'], seq_len=split['seq_len'])
    pred_test = sim(data['test_data_edge_label_index_with_relation'], scorer, batch_size=bs, method='prob',
                    device=device, mask_token=split['mask_token'], seq_len=split['seq_len'])
    return torch.tensor(pred_val, dtype=torch.float32), torch.tensor(pred_test, dtype=torch.float32)


# scores the val/test pairs directly with a pair classifier (alternative to sim()).
# edge_label_index rows are [head, relation, tail].
def score_pairs(edge_label_index, pair_score_fn, batch_size=100000):
    preds = []
    for i in range(0, edge_label_index.shape[1], batch_size):
        h, r, t = edge_label_index[:, i:i + batch_size]
        p = torch.empty(h.shape[0], dtype=torch.float32)
        for rel in r.unique():
            m = r == rel
            p[m] = torch.as_tensor(pair_score_fn(h[m], t[m], rel.item()), dtype=torch.float32).cpu()
        preds.append(p)
    return torch.cat(preds)


#######################################################################################################################
# Training pairs for supervised baselines

# Positive training pairs of one queried relation (training split only) plus negatives sampled exactly
# like the val/test negatives in sentence_generation.py: same number of negatives per head,
# tails drawn from all tails of that relation in the KG, excluding every known KG edge
# (which also excludes all val/test positives).
def training_pairs(split, relation, seed=None):
    seed = split['seed'] if seed is None else seed
    r = split['queried_edge_types'].index(relation)
    edge = split['edge']
    train_txgnn = split['train_txgnn']

    pos = train_txgnn.loc[train_txgnn['relation'] == relation, ["x_index", "y_index", "xy"]].drop_duplicates()
    all_tail_ids = edge.loc[
        (edge['relation'] == relation) & (edge['node_types'] == split['queried_node_types'][r]), 'y_index'].unique()
    neg = negative_samples(pos, edge, all_tail_ids, seed)

    heads = torch.tensor(list(pos['x_index']) + [h for h, _ in neg], dtype=torch.long)
    tails = torch.tensor(list(pos['y_index']) + [t for _, t in neg], dtype=torch.long)
    labels = torch.cat([torch.ones(len(pos), dtype=torch.long), torch.zeros(len(neg), dtype=torch.long)])
    return heads, tails, labels


#######################################################################################################################
# Evaluation

# per-edge-type threshold that maximises F1
def optimal_thresholds(pred, labels, edge_type_flag, edge_flags):
    """Return the threshold that maximises F1 on pred/labels for each edge type."""
    thresholds = []
    for flag in edge_flags:
        mask = (edge_type_flag == flag)
        y_true = labels[mask].numpy()
        y_scores = pred[mask].numpy()
        prec, rec, thresh = precision_recall_curve(y_true, y_scores)
        denom = prec + rec
        f1_arr = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
        # thresh has one fewer element than prec/rec; last prec/rec point is trivial
        best_idx = np.argmax(f1_arr[:-1]) if len(thresh) > 0 else 0
        thresholds.append(float(thresh[best_idx]) if len(thresh) > 0 else 0.5)
    return thresholds


# F1 per edge type; thresholds from val are reused on test
def f1_per_edge_type(pred, labels, edge_type_flag, edge_flags, thresholds=None):
    """Compute F1 per edge type using per-edge-type thresholds.

    If thresholds is None the optimal threshold is derived from pred/labels
    directly (use for val set). Pass val-set thresholds when evaluating the
    test set so the threshold is not fit on the test labels.
    """
    if thresholds is None:
        thresholds = optimal_thresholds(pred, labels, edge_type_flag, edge_flags)
    result = []
    for flag, thr in zip(edge_flags, thresholds):
        mask = (edge_type_flag == flag)
        y_true = labels[mask].numpy()
        y_pred = (pred[mask].numpy() >= thr).astype(int)
        result.append(f1_score(y_true, y_pred, zero_division=0))
    return result, thresholds


# Same metric calls as for NetMedGPT: link_pred_val (AUC, AUPRC), f1_per_edge_type with the threshold chosen
# on val and reused on test, node_level_eval (Recall@100, Precision@100).
# Changes: predictions are passed in (from predict_with_sim() or score_pairs()), and node_level_eval
# receives the baseline scorer (EmbeddingScorer / PairScorer) instead of the transformer.
def evaluate(pred_val, pred_test, data, split, scorer, device):
    edge_flags = split['edge_flags']
    all_param = split['all_param']
    pred_val = torch.as_tensor(pred_val, dtype=torch.float32).cpu()
    pred_test = torch.as_tensor(pred_test, dtype=torch.float32).cpu()

    LP_result_val = link_pred_val(pred_val, data['val_data_edge_label'], data['edge_type_flag_val'], all_param, edge_type=edge_flags)
    f1_val, val_thresholds = f1_per_edge_type(pred_val, data['val_data_edge_label'], data['edge_type_flag_val'], edge_flags)
    hit_k_val, precision_k_val = node_level_eval(data['val_data_edge_label_index_with_relation'], data['val_data_edge_label'],
                                                 split['all_ids_node2'], scorer, device=device,
                                                 mask_token=split['mask_token'], seq_len=split['seq_len'])

    LP_result_test = link_pred_val(pred_test, data['test_data_edge_label'], data['edge_type_flag_test'], all_param, edge_type=edge_flags)
    f1_test, _ = f1_per_edge_type(pred_test, data['test_data_edge_label'], data['edge_type_flag_test'], edge_flags, thresholds=val_thresholds)
    hit_k_test, precision_k_test = node_level_eval(data['test_data_edge_label_index_with_relation'], data['test_data_edge_label'],
                                                   split['all_ids_node2'], scorer, device=device,
                                                   mask_token=split['mask_token'], seq_len=split['seq_len'])

    print(f"val AUPRC {np.array(LP_result_val['auprc']).mean():.4f} | "
          f"test AUC {LP_result_test['auc'].mean():.4f} | test AUPRC {LP_result_test['auprc'].mean():.4f}")
    print(LP_result_test)

    return dict(
        LP_val=LP_result_val, f1_val=f1_val, hit_val=hit_k_val, precision_val=precision_k_val,
        LP_test=LP_result_test, f1_test=f1_test, hit_test=hit_k_test, precision_test=precision_k_test,
        val_thresholds=val_thresholds,
    )


#######################################################################################################################
# Saving results

def save_results(results, split, method, result_root=None):
    queried_edge_types = split['queried_edge_types']
    seed = split['seed']
    if result_root is None:
        result_root = os.path.join(split['data_dir'], 'result', 'new_analysis')

    # val result
    columns = ['auc', 'auprc']
    LP_result_val_df = pd.DataFrame(results['LP_val'], columns=columns)
    LP_result_val_df['f1'] = results['f1_val']
    LP_result_val_df.index = queried_edge_types
    LP_result_val_df['seed'] = seed

    hit_val = pd.DataFrame(results['hit_val'], columns=['hit'])
    precision_val = pd.DataFrame(results['precision_val'], columns=['precision'])
    hit_val.index = queried_edge_types
    precision_val.index = queried_edge_types
    hit_precision_val = pd.concat([hit_val, precision_val], axis=1)
    hit_precision_val['seed'] = seed

    # test result
    LP_result_test_df = pd.DataFrame(results['LP_test'], columns=columns)
    LP_result_test_df['f1'] = results['f1_test']
    LP_result_test_df.index = queried_edge_types
    LP_result_test_df['seed'] = seed

    hit_test = pd.DataFrame(results['hit_test'], columns=['hit'])
    precision_test = pd.DataFrame(results['precision_test'], columns=['precision'])
    hit_test.index = queried_edge_types
    precision_test.index = queried_edge_types
    hit_precision_test = pd.concat([hit_test, precision_test], axis=1)
    hit_precision_test['seed'] = seed

    config_dir = os.path.join(result_root, f"{method}_{split['inference']}_seed{seed}")
    os.makedirs(config_dir, exist_ok=True)

    LP_result_val_df.to_csv(os.path.join(config_dir, "LP_val.csv"), index=True)
    hit_precision_val.to_csv(os.path.join(config_dir, "hit_precision_val.csv"), index=True)
    LP_result_test_df.to_csv(os.path.join(config_dir, "LP_test.csv"), index=True)
    hit_precision_test.to_csv(os.path.join(config_dir, "hit_precision_test.csv"), index=True)
    print(f"Results saved to {config_dir}")

    return LP_result_test_df, hit_precision_test


SPLIT_NAME_STATS = {
    "random_link_split": "random",
    "zero_shot_split": "complex_disease",
}


# appends the test results in the long format of the baseline table used by
# other/statistical_significance.ipynb (columns: Method, Metric, Seed, Split, Task, Metric Name),
# so the walk baselines can be concatenated with the existing baseline results.
# Re-running the same method/split/seed replaces its old rows.
def append_long_format(LP_result_test_df, hit_precision_test, split, method, out_csv=None):
    if out_csv is None:
        out_csv = os.path.join(split['data_dir'], 'result', 'new_analysis', 'walk_baselines_long.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    split_name = SPLIT_NAME_STATS.get(split['inference'], split['inference'])
    metric_cols = [
        (LP_result_test_df, 'auc', 'AUROC'),
        (LP_result_test_df, 'auprc', 'AUPRC'),
        (LP_result_test_df, 'f1', 'F1'),
        (hit_precision_test, 'hit', 'Recall@100'),
        (hit_precision_test, 'precision', 'Precision@100'),
    ]

    rows = []
    for task in split['queried_edge_types']:
        for df, col, metric_name in metric_cols:
            rows.append({
                "Method": method,
                "Metric": float(df.loc[task, col]),
                "Seed": split['seed'],
                "Split": split_name,
                "Task": task,
                "Metric Name": metric_name,
            })
    new = pd.DataFrame(rows)

    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        same_run = (old['Method'] == method) & (old['Seed'] == split['seed']) & (old['Split'] == split_name)
        new = pd.concat([old[~same_run], new], ignore_index=True)

    new.to_csv(out_csv, index=False)
    print(f"Long-format results updated: {out_csv}")
