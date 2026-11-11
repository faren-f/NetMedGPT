"""
PSICHIC (Koh et al., Nature Machine Intelligence 2024) on the drug-protein (DTI) task of the NetMedGPT random split.

PSICHIC is the official code from https://github.com/huankoh/PSICHIC (sequence-based: drug SMILES + protein
sequence, ESM-2 protein features). Its code is not copied here; the cloned folder is passed with --psichic_dir.

Data:
  - SMILES per drug and sequence per protein from dti_inputs.py (run it once before this script).
  - Training pairs : drug_protein edges of the TxGNN train split + negatives sampled exactly like the NetMedGPT
                     val/test negatives (training_pairs() in common.py), only pairs with SMILES and sequence.
  - Val/test pairs : the NetMedGPT drug_protein val/test pairs (positives + negatives) from the corpus cache.
  Proteins longer than --max_protein_len (default 2500 residues, about 0.5% of the drug_protein edges) are not used:
  PSICHIC pools the protein graph with dense residue x residue matrices, which do not fit in GPU memory for very long
  proteins (e.g. titin, 34,350 residues).
  Pairs whose drug has no SMILES (mostly biologics) or whose protein has no (usable) sequence cannot be scored by PSICHIC
  and get score 0. The metrics on the pairs PSICHIC can score are saved as well (LP_test_covered.csv).

Model and training = PSICHIC defaults (config.json, main.py): binary classification head, Adam lr 1e-4,
batch 16, 30 epochs, evaluation after every epoch, best epoch by val ROC-AUC (PSICHIC's own model selection).

Evaluation: same metric functions as NetMedGPT (AUROC, AUPRC, F1 with the val threshold, Recall@100 and
Precision@100 over all gene/protein nodes of the KG). Recall@100 / Precision@100 are computed on the test set only
(val columns are NaN), since every test drug has to be scored against every protein.

Method name in the results: psichic (only the drug_protein task, random_link_split)

Example (from the repository root, in the psichic environment):
    python psichic_baseline.py --seed 1 --gpu 0 --psichic_dir /path/to/PSICHIC
"""

import os
import sys
import glob
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch

from common import load_split, build_corpus, get_device, PairScorer, training_pairs, f1_per_edge_type, save_results, append_long_format
from netmedgpt.utilities import link_pred_val, node_level_eval
from baseline_utils import work_dir, relation_tokens


####### arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID")
parser.add_argument("--seed", required=False, default=1, type=int, help="Random seed / split number")
parser.add_argument("--psichic_dir", required=True, help="path to the cloned PSICHIC repository")
parser.add_argument("--inputs_dir", required=False, default="data/new_analysis/dti_inputs", help="output folder of dti_inputs.py")
parser.add_argument("--epochs", required=False, default=30, type=int, help="PSICHIC default 30")
parser.add_argument("--batch_size", required=False, default=16, type=int, help="PSICHIC default 16")
parser.add_argument("--lrate", required=False, default=1e-4, type=float, help="PSICHIC default 1e-4")
parser.add_argument("--eval_batch_size", required=False, default=64, type=int, help="pairs per forward pass when scoring")
parser.add_argument("--num_workers", required=False, default=8, type=int, help="data loader workers when scoring")
parser.add_argument("--chunk_size", required=False, default=1000, type=int, help="proteins per ESM feature file")
parser.add_argument("--rank", required=False, default=1, type=int, help="1 = Recall@100 / Precision@100 on test (scores test drugs x all proteins), 0 = skip")
parser.add_argument("--max_protein_len", required=False, default=2500, type=int, help="longer proteins are not scored (PSICHIC builds dense residue x residue matrices, GPU memory)")
parser.add_argument("--max_train_pairs", required=False, default=0, type=int, help="only for quick tests, 0 = all training pairs")

INFERENCE = "random_link_split"
TASK = "drug_protein"


def add_psichic_to_path(psichic_dir):
    psichic_dir = os.path.abspath(psichic_dir)
    if psichic_dir not in sys.path:
        sys.path.insert(0, psichic_dir)
    return psichic_dir


#######################################################################################################################
# Split restricted to the drug_protein task

# split and val/test pairs with only the drug_protein relation, so the shared evaluation/saving functions
# produce one row (drug_protein) with the same edge flag as in the full NetMedGPT evaluation
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
# PSICHIC features (seed independent, cached in inputs_dir)

def read_inputs(inputs_dir):
    drugs = pd.read_csv(os.path.join(inputs_dir, 'drug_smiles.csv')).dropna(subset=['smiles'])
    prots = pd.read_csv(os.path.join(inputs_dir, 'protein_sequences.csv')).dropna(subset=['sequence'])
    return dict(zip(drugs['node_index'], drugs['smiles'])), dict(zip(prots['node_index'], prots['sequence']))


# PSICHIC ligand graphs (smiles2graph) for all drugs; SMILES that PSICHIC cannot convert are dropped.
# The cache is completed if drug_smiles.csv has SMILES that are not in it yet.
def ligand_features(smiles, inputs_dir):
    path = os.path.join(inputs_dir, 'psichic_ligands.pt')
    ligands = torch.load(path) if os.path.exists(path) else {}
    todo = sorted(set(smiles.values()) - set(ligands))
    if not todo:
        return ligands
    from utils.ligand_init import smiles2graph
    for s in todo:
        try:
            ligands[s] = smiles2graph(s)
        except Exception as e:
            print(f"ligand skipped ({type(e).__name__}): {s[:60]}")
    torch.save(ligands, path)
    print(f"ligand graphs: {len(ligands)} / {len(set(smiles.values()))} SMILES")
    return ligands


# PSICHIC protein graphs (protein_init: ESM-2 650M residue features + predicted contact map) for all proteins,
# stored in files of chunk_size proteins (node_index order). Returns {chunk file: [node_index, ...]}.
def protein_feature_chunks(seqs, inputs_dir, chunk_size):
    folder = os.path.join(inputs_dir, f'psichic_proteins_{chunk_size}')
    os.makedirs(folder, exist_ok=True)
    from utils.protein_init import protein_init

    ids = sorted(seqs)
    chunks = {}
    for k in range(0, len(ids), chunk_size):
        part = ids[k:k + chunk_size]
        path = os.path.join(folder, f'chunk_{k // chunk_size:03d}.pt')
        if not os.path.exists(path):
            t0 = time.time()
            torch.save(protein_init(sorted(set(seqs[i] for i in part))), path)
            print(f"protein features {k + len(part)}/{len(ids)} ({(time.time() - t0) / 60:.1f} min)")
        chunks[path] = part
    return chunks


# protein graphs of the given node_index values, read from the chunk files ({sequence: features})
def load_proteins(node_ids, seqs, chunks):
    need = set(node_ids)
    prots = {}
    for path, part in chunks.items():
        if need.intersection(part):
            feats = torch.load(path)
            for i in need.intersection(part):
                prots[seqs[i]] = feats[seqs[i]]
    return prots


#######################################################################################################################
# Pairs in PSICHIC format (columns Protein, Ligand, classification_label)

def pair_frame(heads, tails, labels, smiles, seqs, ligands):
    df = pd.DataFrame({'head': np.asarray(heads), 'tail': np.asarray(tails), 'classification_label': np.asarray(labels)})
    df['Ligand'] = df['head'].map(smiles)
    df['Protein'] = df['tail'].map(seqs)
    df['covered'] = df['Ligand'].isin(list(ligands)) & df['Protein'].notna()
    return df


def loader(df, ligands, prots, batch_size, shuffle, num_workers=0):
    from utils.utils import DataLoader
    from utils.dataset import ProteinMoleculeDataset
    dataset = ProteinMoleculeDataset(df.reset_index(drop=True), ligands, prots)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])


# PSICHIC interaction probabilities for the rows of df (same forward call as PSICHIC Trainer.eval)
@torch.no_grad()
def predict(model, df, ligands, prots, batch_size, device, num_workers=0):
    model.eval()
    preds = []
    for data in loader(df, ligands, prots, batch_size, shuffle=False, num_workers=num_workers):
        data = data.to(device)
        _, cls_pred, _, _, _, _, _ = model(
            mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, bond_x=data.mol_edge_attr,
            atom_edge_index=data.mol_edge_index, clique_x=data.clique_x,
            clique_edge_index=data.clique_edge_index, atom2clique_index=data.atom2clique_index,
            residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
            residue_edge_index=data.prot_edge_index, residue_edge_weight=data.prot_edge_weight,
            mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch, clique_batch=data.clique_x_batch)
        preds.append(torch.sigmoid(cls_pred.reshape(-1)).cpu())
    return torch.cat(preds).numpy() if preds else np.zeros(0, dtype=np.float32)


# scores of all pairs in df: PSICHIC probability for covered pairs, 0 for the others
def score_frame(model, df, ligands, prots, batch_size, device, num_workers=0):
    scores = np.zeros(len(df), dtype=np.float32)
    cov = df['covered'].values
    scores[cov] = predict(model, df[cov], ligands, prots, batch_size, device, num_workers)
    return torch.tensor(scores)


#######################################################################################################################
# PSICHIC model and training (as in PSICHIC main.py)

def build_and_train(args, train_df, val_df, ligands, prots, out_dir, device):
    from utils.utils import compute_pna_degrees
    from utils.trainer import Trainer
    from models.net import net

    with open(os.path.join(args.psichic_dir, 'config.json')) as f:
        config = json.load(f)
    config['optimizer']['lrate'] = args.lrate
    config['tasks']['regression_task'] = False
    config['tasks']['classification_task'] = True
    config['tasks']['mclassification_task'] = 0

    train_loader = loader(train_df, ligands, prots, args.batch_size, shuffle=True)
    val_loader = loader(val_df, ligands, prots, args.batch_size, shuffle=False)

    mol_deg, clique_deg, prot_deg = compute_pna_degrees(train_loader)
    p = config['params']
    model = net(mol_deg, prot_deg,
                mol_in_channels=p['mol_in_channels'], prot_in_channels=p['prot_in_channels'],
                prot_evo_channels=p['prot_evo_channels'], hidden_channels=p['hidden_channels'],
                pre_layers=p['pre_layers'], post_layers=p['post_layers'], aggregators=p['aggregators'],
                scalers=p['scalers'], total_layer=p['total_layer'], K=p['K'], heads=p['heads'],
                dropout=p['dropout'], dropout_attn_score=p['dropout_attn_score'],
                regression_head=False, classification_head=True, multiclassification_head=0,
                device=device).to(device)
    model.reset_parameters()
    print(f"PSICHIC parameters: {sum(x.nelement() for x in model.parameters())}")

    # PSICHIC Trainer saves the best epoch (val ROC-AUC) to <out_dir>/save_model_seed<seed>/model.pt
    os.makedirs(os.path.join(out_dir, f'save_model_seed{args.seed}'), exist_ok=True)
    o = config['optimizer']
    engine = Trainer(model=model, lrate=o['lrate'], min_lrate=o['min_lrate'], wdecay=o['weight_decay'],
                     betas=o['betas'], eps=o['eps'], amsgrad=o['amsgrad'], clip=o['clip'],
                     steps_per_epoch=len(train_loader), num_epochs=args.epochs, total_iters=None,
                     warmup_iters=o['warmup_iters'], lr_decay_iters=o['lr_decay_iters'],
                     schedule_lr=o['schedule_lr'], regression_weight=1, classification_weight=1,
                     evaluate_metric='roc', result_path=out_dir, runid=args.seed,
                     skip_test_during_train=True, device=device)
    t0 = time.time()
    engine.train_epoch(train_loader, val_loader=val_loader, test_loader=None, evaluate_epoch=1)
    print(f"training done ({(time.time() - t0) / 60:.1f} min)")

    model.load_state_dict(torch.load(os.path.join(out_dir, f'save_model_seed{args.seed}', 'model.pt')))
    return model


#######################################################################################################################
# Recall@100 / Precision@100: test drugs x all gene/protein nodes

# pair_score_fn(heads, tails, rel) for PairScorer: test drugs x all proteins, scored chunk by chunk
# (one feature file at a time); drugs without SMILES and proteins without sequence get score 0
def ranking_table(model, heads, cands, smiles, seqs, ligands, chunks, args, device):
    heads = heads.unique()
    h_ok = [int(h) for h in heads if smiles.get(int(h)) in ligands]
    table = torch.zeros(len(heads), len(cands), dtype=torch.float32)
    h_pos = {h: i for i, h in enumerate(heads.tolist())}
    c_pos = {c: i for i, c in enumerate(cands.tolist())}

    t0 = time.time()
    for path, part in chunks.items():
        part = [i for i in part if i in c_pos and i in seqs]
        if not part or not h_ok:
            continue
        prots = load_proteins(part, seqs, {path: part})
        hh, tt = np.repeat(h_ok, len(part)), np.tile(part, len(h_ok))
        df = pair_frame(hh, tt, np.zeros(len(hh)), smiles, seqs, ligands)
        s = score_frame(model, df, ligands, prots, args.eval_batch_size, device, args.num_workers)
        table[[h_pos[h] for h in hh], [c_pos[t] for t in tt]] = s
        print(f"  ranking: {os.path.basename(path)} done ({(time.time() - t0) / 60:.1f} min)")
    print(f"ranking: {len(heads)} test drugs ({len(heads) - len(h_ok)} without SMILES) x {len(cands)} proteins")

    max_id = int(max(heads.max(), cands.max())) + 1
    hp = torch.full((max_id,), -1, dtype=torch.long)
    tp = torch.full((max_id,), -1, dtype=torch.long)
    hp[heads] = torch.arange(len(heads))
    tp[cands] = torch.arange(len(cands))

    def pair_score_fn(h, t, rel):
        h, t = h.cpu(), t.cpu()
        out = torch.zeros(len(h), dtype=torch.float32)
        ok = (h < max_id) & (t < max_id)
        ok[ok.clone()] = (hp[h[ok]] >= 0) & (tp[t[ok]] >= 0)
        out[ok] = table[hp[h[ok]], tp[t[ok]]]
        return out

    return pair_score_fn


#######################################################################################################################
# Evaluation (same metric functions as common.evaluate, drug_protein only)

def evaluate_dti(pred_val, pred_test, data, split, scorer, device):
    edge_flags = split['edge_flags']
    all_param = split['all_param']

    LP_val = link_pred_val(pred_val, data['val_data_edge_label'], data['edge_type_flag_val'], all_param, edge_type=edge_flags)
    f1_val, val_thresholds = f1_per_edge_type(pred_val, data['val_data_edge_label'], data['edge_type_flag_val'], edge_flags)
    LP_test = link_pred_val(pred_test, data['test_data_edge_label'], data['edge_type_flag_test'], all_param, edge_type=edge_flags)
    f1_test, _ = f1_per_edge_type(pred_test, data['test_data_edge_label'], data['edge_type_flag_test'], edge_flags, thresholds=val_thresholds)

    if scorer is not None:
        hit_test, precision_test = node_level_eval(data['test_data_edge_label_index_with_relation'], data['test_data_edge_label'],
                                                   split['all_ids_node2'], scorer, device=device,
                                                   mask_token=split['mask_token'], seq_len=split['seq_len'])
    else:
        hit_test, precision_test = [float('nan')], [float('nan')]

    print(f"val AUPRC {np.array(LP_val['auprc']).mean():.4f} | "
          f"test AUC {LP_test['auc'].mean():.4f} | test AUPRC {LP_test['auprc'].mean():.4f} | "
          f"test F1 {f1_test[0]:.4f} | test Recall@100 {hit_test[0]:.4f}")

    return dict(LP_val=LP_val, f1_val=f1_val, hit_val=[float('nan')], precision_val=[float('nan')],
                LP_test=LP_test, f1_test=f1_test, hit_test=hit_test, precision_test=precision_test,
                val_thresholds=val_thresholds)


# AUROC / AUPRC / F1 on the test pairs PSICHIC can score (drug with SMILES and protein with sequence)
def covered_metrics(pred_val, pred_test, val_df, test_df, data, split):
    rows = {}
    for name, pred, df in [('val', pred_val, val_df), ('test', pred_test, test_df)]:
        m = torch.tensor(df['covered'].values)
        lp = link_pred_val(pred[m], data[f'{name}_data_edge_label'][m], data[f'edge_type_flag_{name}'][m],
                           split['all_param'], edge_type=split['edge_flags'])
        rows[name] = (lp, m)
    f1_val, thr = f1_per_edge_type(pred_val[rows['val'][1]], data['val_data_edge_label'][rows['val'][1]],
                                   data['edge_type_flag_val'][rows['val'][1]], split['edge_flags'])
    m = rows['test'][1]
    f1_test, _ = f1_per_edge_type(pred_test[m], data['test_data_edge_label'][m], data['edge_type_flag_test'][m],
                                  split['edge_flags'], thresholds=thr)
    out = rows['test'][0].copy()
    out['f1'] = f1_test
    out.index = [TASK]
    out['n_pairs'] = int(m.sum())
    out['n_pairs_all'] = len(m)
    out['n_pos'] = int(data['test_data_edge_label'][m].sum())
    out['seed'] = split['seed']
    return out


def main():
    args = parser.parse_args()
    device = get_device(args.gpu)
    method = "psichic"
    add_psichic_to_path(args.psichic_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 1. same split and same val/test pairs as NetMedGPT, drug_protein only
    split = load_split(INFERENCE, args.seed)
    data = build_corpus(split)
    split, data, z = task_only(split, data)
    out_dir = work_dir(split, method)

    # 2. SMILES / sequences and PSICHIC features (computed once, reused by all seeds)
    smiles, seqs = read_inputs(args.inputs_dir)
    ligands = ligand_features(smiles, args.inputs_dir)
    chunks = protein_feature_chunks(seqs, args.inputs_dir, args.chunk_size)
    n_all = len(seqs)
    seqs = {i: s for i, s in seqs.items() if len(s) <= args.max_protein_len}
    print(f"proteins scored by PSICHIC: {len(seqs)} / {n_all} (longer than {args.max_protein_len} residues are not scored)")

    # 3. training / val / test pairs
    h, t, y = training_pairs(split, TASK)
    train_df = pair_frame(h.numpy(), t.numpy(), y.numpy(), smiles, seqs, ligands)
    val_idx, test_idx = data['val_data_edge_label_index_with_relation'], data['test_data_edge_label_index_with_relation']
    val_df = pair_frame(val_idx[0].numpy(), val_idx[2].numpy(), data['val_data_edge_label'].numpy(), smiles, seqs, ligands)
    test_df = pair_frame(test_idx[0].numpy(), test_idx[2].numpy(), data['test_data_edge_label'].numpy(), smiles, seqs, ligands)
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"{name}: {len(df)} pairs ({int(df['classification_label'].sum())} positive) | "
              f"PSICHIC can score {int(df['covered'].sum())} ({int(df.loc[df['covered'], 'classification_label'].sum())} positive)")

    train_fit = train_df[train_df['covered']].sample(frac=1, random_state=args.seed)
    if args.max_train_pairs:
        train_fit = train_fit.iloc[:args.max_train_pairs]
    val_fit = val_df[val_df['covered']]
    prots = load_proteins(set(train_fit['tail']) | set(val_df.loc[val_df['covered'], 'tail'])
                          | set(test_df.loc[test_df['covered'], 'tail']), seqs, chunks)

    # 4. PSICHIC training, best epoch by val ROC-AUC
    model = build_and_train(args, train_fit, val_fit, ligands, prots, out_dir, device)

    # 5. scoring of the val/test pairs (0 for pairs PSICHIC cannot score)
    pred_val = score_frame(model, val_df, ligands, prots, args.eval_batch_size, device, args.num_workers)
    pred_test = score_frame(model, test_df, ligands, prots, args.eval_batch_size, device, args.num_workers)
    val_df.assign(score=pred_val.numpy()).drop(columns=['Ligand', 'Protein']).to_csv(os.path.join(out_dir, 'pred_val.csv'), index=False)
    test_df.assign(score=pred_test.numpy()).drop(columns=['Ligand', 'Protein']).to_csv(os.path.join(out_dir, 'pred_test.csv'), index=False)
    del prots

    # 6. Recall@100 / Precision@100 (test drugs x all gene/protein nodes)
    scorer = None
    if args.rank:
        heads = test_idx[0, data['test_data_edge_label'] == 1]
        pair_score_fn = ranking_table(model, heads, split['all_ids_node2'][z], smiles, seqs, ligands, chunks, args, device)
        scorer = PairScorer(pair_score_fn, split['vocab_size'], split['all_ids_node2'], device)

    # 7. metrics and saving (same files as NetMedGPT and the other baselines)
    results = evaluate_dti(pred_val, pred_test, data, split, scorer, device)
    LP_result_test_df, hit_precision_test = save_results(results, split, method)
    if args.rank:
        append_long_format(LP_result_test_df, hit_precision_test, split, method)

    covered = covered_metrics(pred_val, pred_test, val_df, test_df, data, split)
    result_dir = os.path.join(split['data_dir'], 'result', 'new_analysis', f"{method}_{INFERENCE}_seed{args.seed}")
    covered.to_csv(os.path.join(result_dir, 'LP_test_covered.csv'))
    print("test pairs PSICHIC can score:")
    print(covered)


if __name__ == "__main__":
    main()
