"""
Per-pair NetMedGPT scores of the val/test pairs of one task, for paired comparisons with a baseline on exactly the
same pairs (e.g. only the pairs a baseline can score).

For each seed: the saved NetMedGPT checkpoint (data/model_checkpoints/saved_models_evaluation/), the val/test pairs of
the task from the corpus cache (identical to the pairs used in other/evaluate_checkpoints.py), and
  - the score of every val/test pair (same sim() call as other/evaluate_checkpoints.py)
  - Recall@100 / Precision@100 for every test head (same ranking as node_level_eval, all candidate tails of the task),
    so that they can be averaged over any subset of heads.
The mean over all pairs / heads is printed next to the values in data/result/all_checkpoints_evaluation.csv as a check.

Output: data/new_analysis/overlap/netmedgpt_<inference>_<task>_seed<N>.pt

Example (from the repository root, in the netmedgpt environment):
    python export_netmedgpt_scores.py --task drug_effect --gpu 1
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from common import load_split, build_corpus, get_device
from baseline_utils import relation_tokens
from netmedgpt.model import TransformerModel
from netmedgpt.utilities import sim


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID")
parser.add_argument("--inference", required=False, default="random_link_split")
parser.add_argument("--task", required=False, default="drug_effect", help="queried relation, e.g. drug_effect, drug_protein")
parser.add_argument("--seeds", required=False, default="1,2,3,4,5")
parser.add_argument("--k", required=False, default=100, type=int)
parser.add_argument("--out_dir", required=False, default="data/new_analysis/overlap")

CHECKPOINT = {"random_link_split": "data/model_checkpoints/saved_models_evaluation/netmedgpt_randomLinkSplit_seed{}.pt",
              "zero_shot_split": "data/model_checkpoints/saved_models_evaluation/netmedgpt_zeroShot_seed{}.pt"}


# NetMedGPT model with the weights of the checkpoint (as in other/evaluate_checkpoints.py)
def load_model(split, ckpt_path, device):
    nodes, edge = split['nodes'], split['edge']
    hp = split['all_param']['best_hyperparam']
    relation_type = list(edge['relation'].unique())
    entity = list(nodes['node_type'].unique()) + relation_type + ['mask']
    relation_index = edge.loc[edge['relation'].isin(relation_type), ['relation', 'z_index']].drop_duplicates()
    relation_mask_index = pd.concat([relation_index, pd.DataFrame([['mask', split['mask_token']]],
                                                                  columns=['relation', 'z_index'])], ignore_index=True)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt  # two saved formats
    hp = ckpt.get('hyperparams', hp) if 'model_state' in ckpt else hp
    model = TransformerModel(split['vocab_size'], hp['hidden_channels'], hp['nhead'], hp['N_encoder_layers'],
                             split['seq_len'], device=device, feat=split['feat'], nodes=nodes, entity=entity,
                             relation_mask_index=relation_mask_index, pos_emb='fixed').to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, hp['batch_size']


# Recall@k / Precision@k per test head (node_level_eval, without averaging)
@torch.no_grad()
def per_head_recall(model, idx, label, cands, split, device, k, batch_size=256):
    pos = idx[:, label == 1]
    heads = pos[0].unique()
    cands = cands.to(device)
    recall, precision = [], []
    for i in range(0, len(heads), batch_size):
        hb = heads[i:i + batch_size]
        paths = torch.full((len(hb), split['seq_len']), split['mask_token'], dtype=torch.long, device=device)
        paths[:, 0] = hb.to(device)
        paths[:, 1] = int(pos[1, 0])
        logits = model(paths)[:, 2, :][:, cands]
        top = cands[torch.topk(logits, k=k, dim=1).indices].cpu()
        for h, t in zip(hb.tolist(), top):
            true = pos[2, pos[0] == h]
            hit = torch.isin(true, t).sum().item()
            recall.append(hit / len(true))
            precision.append(hit / k)
    return heads, torch.tensor(recall), torch.tensor(precision)


def main():
    args = parser.parse_args()
    device = get_device(args.gpu)
    os.makedirs(args.out_dir, exist_ok=True)
    ref = pd.read_csv("data/result/all_checkpoints_evaluation.csv")

    for seed in [int(s) for s in args.seeds.split(',')]:
        split = load_split(args.inference, seed, load_features=True)
        data = build_corpus(split)
        z = relation_tokens(split)[args.task]
        model, batch_size = load_model(split, CHECKPOINT[args.inference].format(seed), device)

        out = {'task': args.task, 'seed': seed, 'k': args.k}
        for n in ['val', 'test']:
            m = data[f'{n}_data_edge_label_index_with_relation'][1] == z
            idx = data[f'{n}_data_edge_label_index_with_relation'][:, m]
            out[f'{n}_idx'], out[f'{n}_label'] = idx, data[f'{n}_data_edge_label'][m]
            out[f'{n}_score'] = torch.tensor(sim(idx, model, batch_size=batch_size, method='prob', device=device,
                                                 mask_token=split['mask_token'], seq_len=split['seq_len']))
        out['heads'], out['recall'], out['precision'] = per_head_recall(
            model, out['test_idx'], out['test_label'], split['all_ids_node2'][z], split, device, args.k)
        path = os.path.join(args.out_dir, f"netmedgpt_{args.inference}_{args.task}_seed{seed}.pt")
        torch.save(out, path)

        r = ref[(ref['inference'] == args.inference) & (ref['seed'] == seed) & (ref['edge_type'] == args.task)]
        y, s = out['test_label'].numpy(), out['test_score'].numpy()
        print(f"seed {seed}: AUROC {roc_auc_score(y, s):.4f} (csv {r['auc'].item():.4f}) | "
              f"AUPRC {average_precision_score(y, s):.4f} (csv {r['auprc'].item():.4f}) | "
              f"Recall@{args.k} {out['recall'].mean():.4f} (csv {r['recall@100'].item():.4f}) -> {path}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
