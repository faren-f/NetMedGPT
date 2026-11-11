"""
DeepADR (Wang et al., bioRxiv 10.1101/2025.07.29.667409; https://github.com/Wjt777/DeepADR) on the drug-effect (ADR)
task of the NetMedGPT random split.

DeepADR is used from the cloned repository (passed with --deepadr_dir): the DeepADR model class and its
training / prediction functions from train.py (train_one_epoch, collect_predictions, build_model).

Inputs per drug-ADR pair (as in DeepADR):
  - drug structure : MoLFormer embedding of the drug SMILES (adr_inputs.py)
  - drug targets   : multi-hot vector over all proteins that have a drug_protein edge in the TxGNN train split
  - ADR semantics  : BioBERT [CLS] embedding of the effect/phenotype name (adr_inputs.py)
  Drugs without SMILES (mostly biologics) get a zero MoLFormer vector, so DeepADR still scores them from their
  targets. The metrics on the pairs whose drug has a SMILES are saved as well (LP_test_covered.csv).

Data:
  - Training pairs : drug_effect edges of the TxGNN train split + negatives sampled exactly like the NetMedGPT
                     val/test negatives (training_pairs() in common.py), about 1:1 as in DeepADR.
  - Val/test pairs : the NetMedGPT drug_effect val/test pairs (positives + negatives) from the corpus cache.

Training = DeepADR train.py defaults for classification: AdamW lr 1e-3, weight decay 1e-4, batch 32,
25 epochs, BCE loss, gradient clipping 1.0, best epoch by val loss (DeepADR's own model selection,
computed on the NetMedGPT val pairs).

Evaluation: same functions and files as NetMedGPT and the other baselines (AUROC, AUPRC, F1 with the val threshold,
Recall@100 / Precision@100 over all effect/phenotype nodes).

Method name in the results: deepadr (only the drug_effect task, random_link_split)

Example (from the repository root, in the deepadr environment):
    python deepadr_baseline.py --seed 1 --gpu 0 --deepadr_dir /path/to/DeepADR
"""

import os
import sys
import time
import types
import random
import argparse
import importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from common import load_split, build_corpus, get_device, PairScorer, score_pairs, training_pairs, f1_per_edge_type
from netmedgpt.utilities import link_pred_val
from baseline_utils import finish, work_dir, relation_tokens, in_dir


####### arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID")
parser.add_argument("--seed", required=False, default=1, type=int, help="Random seed / split number")
parser.add_argument("--deepadr_dir", required=True, help="path to the cloned DeepADR repository")
parser.add_argument("--inputs_dir", required=False, default="data/new_analysis/adr_inputs", help="output folder of adr_inputs.py")
parser.add_argument("--epochs", required=False, default=25, type=int, help="DeepADR default 25")
parser.add_argument("--batch_size", required=False, default=32, type=int, help="DeepADR default 32")
parser.add_argument("--lr", required=False, default=1e-3, type=float, help="DeepADR default 1e-3")
parser.add_argument("--weight_decay", required=False, default=1e-4, type=float, help="DeepADR default 1e-4")
parser.add_argument("--kan_hidden_dim", required=False, default=16, type=int, help="DeepADR default 16")
parser.add_argument("--eval_batch_size", required=False, default=16384, type=int, help="pairs per forward pass when scoring")
parser.add_argument("--max_train_pairs", required=False, default=0, type=int, help="only for quick tests, 0 = all training pairs")

INFERENCE = "random_link_split"
TASK = "drug_effect"


# DeepADR package (model) and its train.py (training functions); train.py is loaded under its own name
# because the NetMedGPT repository has a train.py as well
def load_deepadr(deepadr_dir):
    deepadr_dir = os.path.abspath(deepadr_dir)
    if deepadr_dir not in sys.path:
        sys.path.append(deepadr_dir)
    spec = importlib.util.spec_from_file_location("deepadr_train", os.path.join(deepadr_dir, "train.py"))
    deepadr_train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deepadr_train)
    return deepadr_train


#######################################################################################################################
# Split restricted to the drug_effect task

# split and val/test pairs with only the drug_effect relation, so the shared evaluation/saving functions
# produce one row (drug_effect) with the same edge flag as in the full NetMedGPT evaluation
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


#######################################################################################################################
# DeepADR features: one row per KG drug / effect node

# feature matrices + node_index -> row lookups:
#   mol    (n_drugs, 768)      MoLFormer, zero for drugs without SMILES
#   target (n_drugs, n_prots)  multi-hot over the proteins with a training drug_protein edge
#   adr    (n_effects, 768)    BioBERT [CLS]
def build_features(split, inputs_dir):
    nodes = split['nodes']
    vocab = int(nodes['node_index'].max()) + 1
    drug_ids = nodes.loc[nodes['node_type'] == 'drug', 'node_index'].values
    drug_row = torch.full((vocab,), -1, dtype=torch.long)
    drug_row[drug_ids] = torch.arange(len(drug_ids))

    m = torch.load(os.path.join(inputs_dir, 'drug_molformer.pt'))
    mol = torch.zeros(len(drug_ids), m['emb'].shape[1])
    mol[drug_row[m['node_index']]] = m['emb']
    has_smiles = torch.zeros(vocab, dtype=torch.bool)
    has_smiles[m['node_index']] = True

    # training drug_protein edges in both orientations -> (drug, protein)
    tr = split['train_txgnn']
    tr = tr.loc[tr['relation'] == 'drug_protein', ['x_index', 'y_index']].values
    is_drug = torch.zeros(vocab, dtype=torch.bool)
    is_drug[drug_ids] = True
    x, y = torch.tensor(tr[:, 0]), torch.tensor(tr[:, 1])
    d = torch.where(is_drug[x], x, y)
    p = torch.where(is_drug[x], y, x)
    prots = p.unique()
    prot_col = torch.full((vocab,), -1, dtype=torch.long)
    prot_col[prots] = torch.arange(len(prots))
    target = torch.zeros(len(drug_ids), len(prots))
    target[drug_row[d], prot_col[p]] = 1.0

    a = torch.load(os.path.join(inputs_dir, 'phenotype_biobert.pt'))
    adr_row = torch.full((vocab,), -1, dtype=torch.long)
    adr_row[a['node_index']] = torch.arange(len(a['node_index']))

    print(f"features: {len(drug_ids)} drugs ({int(has_smiles.sum())} with SMILES), "
          f"{len(prots)} target proteins, {len(a['node_index'])} ADR terms")
    return dict(mol=mol, target=target, adr=a['emb'].float(), drug_row=drug_row, adr_row=adr_row,
                has_smiles=has_smiles)


# pairs in the (mol, target, adr, label) format of DeepADR's DrugADRDataset
class PairDataset(Dataset):
    def __init__(self, feats, heads, tails, labels):
        self.f = feats
        self.d = feats['drug_row'][torch.as_tensor(heads)]
        self.a = feats['adr_row'][torch.as_tensor(tails)]
        self.y = torch.as_tensor(labels, dtype=torch.float32).view(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.f['mol'][self.d[i]], self.f['target'][self.d[i]], self.f['adr'][self.a[i]], self.y[i]


#######################################################################################################################
# Scoring

# DeepADR probabilities for (head, tail) pairs, computed in batches directly from the feature matrices
@torch.no_grad()
def pair_probs(model, feats, heads, tails, device, batch_size):
    model.eval()
    d = feats['drug_row'][torch.as_tensor(heads).cpu()]
    a = feats['adr_row'][torch.as_tensor(tails).cpu()]
    out = []
    for i in range(0, len(d), batch_size):
        di, ai = d[i:i + batch_size], a[i:i + batch_size]
        logits = model(feats['mol'][di].to(device), feats['target'][di].to(device), feats['adr'][ai].to(device))
        out.append(torch.sigmoid(logits).float().cpu())
    return torch.cat(out) if out else torch.zeros(0)


# AUROC / AUPRC / F1 on the pairs whose drug has a SMILES (val threshold reused on test)
def covered_metrics(pred_val, pred_test, data, split, feats):
    masks = {n: feats['has_smiles'][data[f'{n}_data_edge_label_index_with_relation'][0]] for n in ['val', 'test']}
    _, thr = f1_per_edge_type(pred_val[masks['val']], data['val_data_edge_label'][masks['val']],
                              data['edge_type_flag_val'][masks['val']], split['edge_flags'])
    m = masks['test']
    out = link_pred_val(pred_test[m], data['test_data_edge_label'][m], data['edge_type_flag_test'][m],
                        split['all_param'], edge_type=split['edge_flags'])
    out['f1'], _ = f1_per_edge_type(pred_test[m], data['test_data_edge_label'][m], data['edge_type_flag_test'][m],
                                    split['edge_flags'], thresholds=thr)
    out.index = [TASK]
    out['n_pairs'] = int(m.sum())
    out['n_pairs_all'] = len(m)
    out['n_pos'] = int(data['test_data_edge_label'][m].sum())
    out['seed'] = split['seed']
    return out


def main():
    args = parser.parse_args()
    device = get_device(args.gpu)
    args.device = str(device)
    method = "deepadr"
    deepadr_train = load_deepadr(args.deepadr_dir)
    deepadr_train.set_seed(args.seed)

    # 1. same split and same val/test pairs as NetMedGPT, drug_effect only
    split = load_split(INFERENCE, args.seed)
    data = build_corpus(split)
    split, data, z = task_only(split, data)
    out_dir = work_dir(split, method)
    feats = build_features(split, args.inputs_dir)

    # 2. training / val pairs
    h, t, y = training_pairs(split, TASK)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(args.seed))
    if args.max_train_pairs:
        perm = perm[:args.max_train_pairs]
    train_set = PairDataset(feats, h[perm], t[perm], y[perm])
    val_idx = data['val_data_edge_label_index_with_relation']
    val_set = PairDataset(feats, val_idx[0], val_idx[2], data['val_data_edge_label'])
    print(f"train: {len(train_set)} pairs ({int(train_set.y.sum())} positive) | val: {len(val_set)} pairs")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # 3. DeepADR model and training loop (as in DeepADR train.py main(), classification task).
    # The KAN layer (pykan) writes checkpoint files to ./model, so the model is built inside the work folder.
    shapes = types.SimpleNamespace(drug_molformer=feats['mol'], drug_target=feats['target'], adr_features=feats['adr'])
    with in_dir(out_dir):
        model = deepadr_train.build_model(shapes, args)
    # pykan speed mode: skips the symbolic branch of the KAN layer, which is masked to zero in DeepADR (identical
    # outputs and gradients) but evaluated element by element in Python (about 30x slower per batch)
    model.kan_layer.speed()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss, best_epoch = float("inf"), -1
    best_path = os.path.join(out_dir, "deepadr_classification_best.pt")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = deepadr_train.train_one_epoch(model, train_loader, criterion, optimizer, args.device, "classification")
        val_labels, val_logits = deepadr_train.collect_predictions(model, val_loader, args.device)
        val_loss = criterion(torch.tensor(val_logits, dtype=torch.float32),
                             torch.tensor(val_labels, dtype=torch.float32)).item()
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), best_path)
        print(f"Epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"({(time.time() - t0) / 60:.1f} min)")
    model.load_state_dict(torch.load(best_path, map_location=args.device))
    print(f"best epoch {best_epoch}: val loss {best_val_loss:.4f}")

    # 4. scoring of the val/test pairs and of all candidates (Recall@100), same metrics and files as NetMedGPT
    def pair_score_fn(heads, tails, rel):
        return pair_probs(model, feats, heads, tails, args.device, args.eval_batch_size)

    scorer = PairScorer(pair_score_fn, split['vocab_size'], split['all_ids_node2'], device)
    pred_val = score_pairs(data['val_data_edge_label_index_with_relation'], pair_score_fn)
    pred_test = score_pairs(data['test_data_edge_label_index_with_relation'], pair_score_fn)
    finish(split, data, pred_val, pred_test, scorer, device, method)

    covered = covered_metrics(pred_val, pred_test, data, split, feats)
    result_dir = os.path.join(split['data_dir'], 'result', 'new_analysis', f"{method}_{INFERENCE}_seed{args.seed}")
    covered.to_csv(os.path.join(result_dir, 'LP_test_covered.csv'))
    print("test pairs whose drug has a SMILES:")
    print(covered)


if __name__ == "__main__":
    main()
