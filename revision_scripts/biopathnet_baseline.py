"""
BioPathNet (Hu et al., Nature Biomedical Engineering 2026) on the same TxGNN splits as NetMedGPT.

BioPathNet is the official code from https://github.com/emyyue/BioPathNet (NBFNet-based, path representation
learning). Its code is not copied here; the cloned folder is passed with --biopathnet_dir.

Data given to BioPathNet (BioPathNet's own file format, written by this script):
  train1.txt : background graph (BRG) = all training edges of the non-supervised relations
  train2.txt : supervision edges     = training edges of the queried relations (+ the drug-disease relations,
                                        as in the BioPathNet drug repurposing setup)
  valid.txt  : val positives of the queried relations  (not used for message passing)
  test.txt   : test positives of the queried relations (not used for message passing)
All edges come from the TxGNN train split only (same training graph as NetMedGPT); reverse (rev_*) relations
are dropped because BioPathNet adds inverse edges itself (as in BioPathNet reproduce/primekg/01_get_datasplit.py).

Model hyperparameters = BioPathNet config/primekg/cardiovascular.yaml (6 layers x 32, distmult + PNA,
BCE with 64 strict, type-aware negatives, adversarial temperature 0.5, Adam lr 1e-3, batch 16, 8 epochs).
The epoch with the best mean val AUPRC on the NetMedGPT val pairs is kept.

Evaluation: the val/test pairs (positives + negatives), the candidate tails for Recall@100 / Precision@100 and the
metrics are exactly the ones used for NetMedGPT and the walk baselines (common.py).

Method name in the results: biopathnet

Example (from the repository root, in the biopathnet environment):
    python biopathnet_baseline.py --inference random_link_split --seed 1 --gpu 0 --biopathnet_dir /path/to/BioPathNet
"""

import os
import sys
import time
import copy
import random
import argparse
import numpy as np
import pandas as pd
import torch
import easydict

from common import load_split, build_corpus, get_device, PairScorer, score_pairs
from netmedgpt.utilities import link_pred_val
from baseline_utils import finish, work_dir, relation_tokens


####### arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID")
parser.add_argument("--inference", required=False, default="random_link_split", help="random_link_split, zero_shot_split, rare_disease or a disease area")
parser.add_argument("--seed", required=False, default=1, type=int, help="Random seed / split number")
parser.add_argument("--biopathnet_dir", required=True, help="path to the cloned BioPathNet repository")
parser.add_argument("--num_epoch", required=False, default=8, type=int, help="BioPathNet default 8")
parser.add_argument("--batch_size", required=False, default=16, type=int, help="BioPathNet default 16")
parser.add_argument("--lr", required=False, default=1e-3, type=float, help="BioPathNet default 1e-3")
parser.add_argument("--num_layers", required=False, default=6, type=int, help="BioPathNet default 6")
parser.add_argument("--hidden_dim", required=False, default=32, type=int, help="BioPathNet default 32")
parser.add_argument("--eval_batch_size", required=False, default=16, type=int, help="heads scored per forward pass")
parser.add_argument("--max_steps_per_epoch", required=False, default=0, type=int, help="only for quick tests, 0 = full epochs")

# drug-disease relations are always supervised, as in the BioPathNet drug repurposing setup
DRUG_DISEASE = ['indication', 'contraindication', 'off-label use']


def add_biopathnet_to_path(biopathnet_dir):
    biopathnet_dir = os.path.abspath(biopathnet_dir)
    if biopathnet_dir not in sys.path:
        sys.path.insert(0, biopathnet_dir)
    return biopathnet_dir


#######################################################################################################################
# Input files in BioPathNet format

# writes train1/train2/valid/test (tab-separated head, relation, tail) and entity_types.txt into out_dir.
# Node tokens are the KG node_index values, relation tokens the KG relation names.
def write_biopathnet_files(split, data, out_dir):
    supervised = list(dict.fromkeys(split['queried_edge_types'] + DRUG_DISEASE))

    train = split['train_txgnn']
    train = train.loc[~train['relation'].str.startswith('rev_'), ['x_index', 'relation', 'y_index']].drop_duplicates()
    is_sup = train['relation'].isin(supervised)
    train1 = train[~is_sup].sample(frac=1, random_state=split['seed'])
    train2 = train[is_sup].sample(frac=1, random_state=split['seed'])

    # val/test positives = the positive NetMedGPT val/test pairs, relation token -> relation name
    token2rel = {v: k for k, v in relation_tokens(split).items()}

    def positives(name):
        idx = data[f'{name}_data_edge_label_index_with_relation']
        idx = idx[:, data[f'{name}_data_edge_label'] == 1]
        return pd.DataFrame({'x_index': idx[0].numpy(),
                             'relation': [token2rel[int(r)] for r in idx[1]],
                             'y_index': idx[2].numpy()})

    files = {'train1.txt': train1, 'train2.txt': train2, 'valid.txt': positives('val'), 'test.txt': positives('test')}
    for name, df in files.items():
        df[['x_index', 'relation', 'y_index']].to_csv(os.path.join(out_dir, name), sep='\t', index=False, header=False)
        print(f"{name}: {len(df)} triplets")

    # node type for every KG node (BioPathNet only reads the nodes that occur in the triplet files)
    nodes = split['nodes']
    types = pd.DataFrame({'idx': nodes['node_index'], 'type_id': nodes['node_type'].astype('category').cat.codes})
    types.to_csv(os.path.join(out_dir, 'entity_types.txt'), sep='\t', index=False, header=False)
    print(f"supervised relations: {supervised}")


#######################################################################################################################
# BioPathNet model

# config/primekg/cardiovascular.yaml of the BioPathNet repository, with the data path and GPU filled in
def biopathnet_config(args, data_path):
    return easydict.EasyDict({
        'dataset': {'class': 'biomedical', 'path': data_path, 'include_factgraph': True},
        'task': {
            'class': 'KnowledgeGraphCompletionBiomed',
            'model': {
                'class': 'NBFNet',
                'input_dim': args.hidden_dim,
                'hidden_dims': [args.hidden_dim] * args.num_layers,
                'message_func': 'distmult',
                'aggregate_func': 'pna',
                'short_cut': True,
                'layer_norm': True,
                'dependent': True,
                'symmetric': False,
            },
            'criterion': 'bce',
            'num_negative': 64,
            'strict_negative': True,
            'adversarial_temperature': 0.5,
            'sample_weight': False,
            'heterogeneous_negative': True,
            'heterogeneous_evaluation': True,
            'full_batch_eval': True,
        },
        'optimizer': {'class': 'Adam', 'lr': args.lr},
        'engine': {'gpus': [args.gpu], 'batch_size': args.batch_size},
    })


# scores of all candidate tails for each head: (len(heads), len(cands)) matrix of BioPathNet probabilities.
# One NBFNet pass per batch of heads gives the scores of every node, so all candidates are scored at once.
# Heads or tails that do not occur in the BioPathNet graph files get score 0 (ranked last).
@torch.no_grad()
def score_matrix(solver, dataset, heads, cands, rel_name, batch_size):
    task = solver.model
    model = task.model
    model.eval()
    device = solver.device
    vocab = dataset.inv_entity_vocab
    r = dataset.inv_relation_vocab[rel_name]

    h_id = torch.tensor([vocab.get(str(int(h)), -1) for h in heads], dtype=torch.long)
    t_id = torch.tensor([vocab.get(str(int(t)), -1) for t in cands], dtype=torch.long)
    h_ok = (h_id >= 0).nonzero(as_tuple=True)[0]
    t_ok = (t_id >= 0).nonzero(as_tuple=True)[0]
    t_index_all = t_id[t_ok].to(device)

    scores = torch.zeros(len(heads), len(cands), dtype=torch.float32)
    for i in range(0, len(h_ok), batch_size):
        rows = h_ok[i:i + batch_size]
        h_index = h_id[rows].to(device).unsqueeze(1).expand(-1, len(t_index_all))
        t_index = t_index_all.unsqueeze(0).expand(len(rows), -1)
        r_index = torch.full_like(h_index, r)
        s = torch.sigmoid(model(task.fact_graph, h_index, t_index, r_index)).float().cpu()
        scores[rows.unsqueeze(1), t_ok.unsqueeze(0)] = s

    n_missing = (len(heads) - len(h_ok), len(cands) - len(t_ok))
    return scores, n_missing


# score matrices for every queried relation: unique heads of the val/test pairs x all candidate tails
# (all_ids_node2, the same candidates as for NetMedGPT). Returns {rel_token: (head_ids, cand_ids, scores)}.
def score_tables(solver, dataset, split, data, names, batch_size):
    rel_tok = relation_tokens(split)
    tables = {}
    for rel_name, tok in rel_tok.items():
        heads = torch.cat([data[f'{n}_data_edge_label_index_with_relation'][0][
                               data[f'{n}_data_edge_label_index_with_relation'][1] == tok] for n in names]).unique()
        cands = split['all_ids_node2'][tok]
        scores, (mh, mt) = score_matrix(solver, dataset, heads, cands, rel_name, batch_size)
        tables[tok] = (heads, cands, scores)
        print(f"{rel_name}: {len(heads)} heads x {len(cands)} candidates "
              f"(not in BioPathNet graph: {mh} heads, {mt} candidates)")
    return tables


# pair_score_fn(heads, tails, rel_token) for score_pairs() and PairScorer, read from the score tables
def lookup_fn(tables):
    maps = {}
    for tok, (heads, cands, scores) in tables.items():
        h_pos = torch.full((int(max(heads.max(), cands.max())) + 1,), -1, dtype=torch.long)
        t_pos = h_pos.clone()
        h_pos[heads] = torch.arange(len(heads))
        t_pos[cands] = torch.arange(len(cands))
        maps[tok] = (h_pos, t_pos, scores)

    def pair_score_fn(heads, tails, rel):
        h_pos, t_pos, scores = maps[rel]
        heads, tails = heads.cpu(), tails.cpu()
        out = torch.zeros(len(heads), dtype=torch.float32)
        ok = (heads < len(h_pos)) & (tails < len(t_pos))
        hp = torch.full_like(heads, -1)
        tp = torch.full_like(tails, -1)
        hp[ok], tp[ok] = h_pos[heads[ok]], t_pos[tails[ok]]
        ok = (hp >= 0) & (tp >= 0)
        out[ok] = scores[hp[ok], tp[ok]]
        return out

    return pair_score_fn


# mean val AUPRC over the queried relations (model selection between epochs)
def val_auprc(solver, dataset, split, data, batch_size):
    pair_score_fn = lookup_fn(score_tables(solver, dataset, split, data, ['val'], batch_size))
    pred_val = score_pairs(data['val_data_edge_label_index_with_relation'], pair_score_fn)
    res = link_pred_val(pred_val, data['val_data_edge_label'], data['edge_type_flag_val'], split['all_param'],
                        edge_type=split['edge_flags'])
    return float(np.mean(res['auprc']))


def main():
    args = parser.parse_args()
    device = get_device(args.gpu)
    method = "biopathnet"

    # 1. same split and same val/test pairs as NetMedGPT
    split = load_split(args.inference, args.seed)
    data = build_corpus(split)
    out_dir = work_dir(split, method)
    write_biopathnet_files(split, data, out_dir)

    # 2. BioPathNet dataset, model and trainer (official code)
    add_biopathnet_to_path(args.biopathnet_dir)
    from torchdrug import core
    from biopathnet import dataset as bpn_dataset, layer, model, task, util  # registers the BioPathNet classes

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = biopathnet_config(args, out_dir)
    bpn_data = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, bpn_data)
    if args.max_steps_per_epoch:
        solver.train_set = torch.utils.data.Subset(solver.train_set, range(args.max_steps_per_epoch * args.batch_size))

    # 3. training, keeping the epoch with the best val AUPRC
    best, best_epoch, best_state = -1.0, -1, None
    for epoch in range(args.num_epoch):
        t0 = time.time()
        solver.model.split = "train"
        solver.train(num_epoch=1)
        auprc = val_auprc(solver, bpn_data, split, data, args.eval_batch_size)
        print(f"epoch {epoch + 1}: val AUPRC {auprc:.4f} ({(time.time() - t0) / 60:.1f} min)")
        if auprc > best:
            best, best_epoch = auprc, epoch + 1
            best_state = copy.deepcopy({k: v.cpu() for k, v in solver.model.model.state_dict().items()})
    solver.model.model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "best_model.pt"))
    print(f"best epoch {best_epoch}: val AUPRC {best:.4f}")

    # 4. scoring of the val/test pairs and of all candidates (Recall@100), same metrics and output files as NetMedGPT
    tables = score_tables(solver, bpn_data, split, data, ['val', 'test'], args.eval_batch_size)
    pair_score_fn = lookup_fn(tables)
    scorer = PairScorer(pair_score_fn, split['vocab_size'], split['all_ids_node2'], device)
    pred_val = score_pairs(data['val_data_edge_label_index_with_relation'], pair_score_fn)
    pred_test = score_pairs(data['test_data_edge_label_index_with_relation'], pair_score_fn)

    finish(split, data, pred_val, pred_test, scorer, device, method)


if __name__ == "__main__":
    main()
