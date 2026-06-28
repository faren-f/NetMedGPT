"""
subnetwork_generator.py

Generates biological subnetworks using metapath generation with frequency consensus filtering:
  - N_METAPATH paths are generated in a single batched forward pass per position
    (~50x fewer forward passes than the original sequential approach)
  - Only triplets appearing in >= FREQ_THRESHOLD of paths are retained

Usage:
    python subnetwork_generator.py \
        --gpu 0 \
        --head_index 14016 \
        --head_type drug \
        --tail_type gene/protein \
        --relation_type drug_protein \
        --N_top 5

Output:
    data/user_response/subnetwork.csv
"""


import os
import json
import time
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from netmedgpt.model import TransformerModel, Classifier
from config import set_seed, print_reproducibility_settings

####### arg parser
parser = argparse.ArgumentParser(description='NetMedGPT subnetwork generator (optimized)')
parser.add_argument("--gpu",            required=False, default=0,               help="GPU device ID")
parser.add_argument("--head_index",     required=False, default=[14016],          help="node_index of head node(s)", nargs='+', type=int)
parser.add_argument("--head_type",      required=False, default="disease",        help="type of the head node")
parser.add_argument("--tail_type",      required=False, default="drug",           help="type of the tail node")
parser.add_argument("--relation_type",  required=False, default="indication",     help="relation type between head and tail")
parser.add_argument("--N_top",          required=False, default=100,  type=int,   help="number of top tail predictions per head")
parser.add_argument("--N_metapath",     required=False, default=200,  type=int,   help="number of paths per direction")
parser.add_argument("--temperature",    required=False, default=1.0,  type=float, help="sampling temperature")
parser.add_argument("--freq_threshold", required=False, default=0.15, type=float, help="minimum triplet frequency for consensus")
parser.add_argument("--lp_threshold",   required=False, default=0.8,  type=float, help="minimum LP indication score to process a tail candidate")

args = parser.parse_args()

SEED           = 1
TEMPERATURE    = args.temperature
N_METAPATH     = args.N_metapath
FREQ_THRESHOLD = args.freq_threshold
LP_THRESHOLD   = args.lp_threshold

set_seed(SEED)
print_reproducibility_settings(SEED)

device        = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
head_idx      = args.head_index
head_type     = args.head_type
tail_type     = args.tail_type
relation_type = args.relation_type
N_top         = args.N_top

print(f"Device        : {device}")
print(f"Head nodes    : {head_idx}")
print(f"Relation      : {relation_type}  ({head_type} → {tail_type})")
print(f"N_top         : {N_top}")
print(f"N_metapath    : {N_METAPATH}  per direction")
print(f"Temperature   : {TEMPERATURE}")
print(f"Freq threshold: {FREQ_THRESHOLD}")
print(f"LP threshold  : {LP_THRESHOLD}")

#########################
# load parameters and data
with open("data/parameters.json", 'r') as file:
    all_param = json.load(file)

model_dir  = os.path.join(all_param['files']['data_dir'], 'model_checkpoints')
data_dir   = all_param['files']['data_dir']
output_dir = os.path.join(all_param['files']['data_dir'], 'user_response', 'subnetworks')
os.makedirs(output_dir, exist_ok=True)

checkpoint_path    = os.path.join(model_dir, "netmedgpt.pt")
checkpoint_path_LP = os.path.join(model_dir, "netmedgpt_LP_indication.pt")

nodes = pd.read_csv(os.path.join(data_dir, 'KG/nodes_preprocessed.csv'), sep=',')
edge  = pd.read_csv(os.path.join(data_dir, 'KG/edges_preprocessed.csv'))
feat  = torch.load(os.path.join(data_dir, 'KG/embeddings_preprocessedKG.pt'))

# identify edge indices and concat with node indices
index_relations = edge[['relation', 'z_index', 'relation']].drop_duplicates()
index_relations.columns = ['node_type', 'node_index', 'node_name']
node_edge_indices = pd.concat([nodes, index_relations])
mask_token_id     = edge['z_index'].max() + 1
relation_token_id = edge.loc[edge['relation'] == relation_type, 'z_index'].drop_duplicates().values.item()

# for model instantiation
all_relation_type   = list(edge['relation'].unique())
node_types          = list(nodes['node_type'].unique())
entity              = node_types + all_relation_type + ['mask']
relation_index      = edge.loc[edge['relation'].isin(all_relation_type), ['relation', 'z_index']].drop_duplicates()
mask_row            = pd.DataFrame([['mask', mask_token_id]], columns=['relation', 'z_index'])
relation_mask_index = pd.concat([relation_index, mask_row], ignore_index=True)
vocab_size          = mask_token_id + 1

#########################
# load model
print('\nLoading model...')
checkpoint = torch.load(checkpoint_path, map_location=device)
param      = checkpoint['parameters']
state_dict = checkpoint['model_state_dict']

model = TransformerModel(
    vocab_size,
    param['hidden_channels'],
    param['nhead'],
    param['N_encoder_layers'],
    (param['walk_length'] * 2) - 1,
    device=device,
    feat=feat,
    nodes=nodes,
    entity=entity,
    relation_mask_index=relation_mask_index,
    pos_emb='fixed',
).to(device)
model.load_state_dict(state_dict)
model.eval()
print(f'Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}')

#########################
# load LP indication classifier
print('Loading LP indication classifier...')
checkpoint_LP = torch.load(checkpoint_path_LP, map_location=device)
param_LP      = checkpoint_LP['parameters']
classifier    = Classifier(param_LP['hidden_channels_encoder'], param_LP['d_classifier'], device)
classifier.load_state_dict(checkpoint_LP['classifier_state_dict'])
classifier.eval()

#########################
# lookup maps and constants
seq_len        = param['walk_length'] * 2 - 1
n_mask         = seq_len - 3  # positions after [head, relation, tail] are masked

index_name_map = dict(zip(node_edge_indices['node_index'], node_edge_indices['node_name']))
index_type_map = dict(zip(node_edge_indices['node_index'], node_edge_indices['node_type']))
relation_edges = set(edge.loc[
    (edge['relation'] == relation_type) & (edge['node_types'] == f"{head_type}|{tail_type}"), 'xy'
])

tail_ids    = nodes.loc[nodes['node_type'] == tail_type, 'node_index'].values
tail_tensor = torch.tensor(tail_ids.astype(int)).to(device)
N_actual    = min(N_top, len(tail_ids))

##############################################################################################################
# Step 1: top-k tail prediction for all head nodes (single batched pass)
sentence = torch.full((len(head_idx), seq_len), mask_token_id, dtype=torch.long)
sentence[:, 0] = torch.tensor(head_idx, dtype=torch.long)
sentence[:, 1] = relation_token_id

dataloader    = DataLoader(TensorDataset(sentence), batch_size=512, shuffle=False)
top_tails_all = []

for batch in tqdm(dataloader, desc='Top-k tails'):
    input_batch = batch[0].to(device)
    with torch.inference_mode():
        output = model(input_batch)
        logits = output[:, 2, tail_tensor]
        probs  = F.softmax(logits, dim=1)
        _, top_idx = torch.topk(probs, N_actual, dim=1)
        top_tails_all.append(tail_tensor[top_idx].cpu())

top_tails_all = torch.cat(top_tails_all, dim=0)  # (n_heads, N_actual)

##############################################################################################################
def metapath_generation_batched(walk_input, n_metapath, mask_token, model, T=1.0):
    """
    Generate n_metapath independent paths from a seed walk using a single batched
    forward pass per masked position (~50x fewer passes than the sequential approach).
    """
    walks = walk_input.expand(n_metapath, -1).clone()          # (n_metapath, L)
    L = walks.shape[1]
    probs        = torch.ones(n_metapath, L, dtype=torch.double, device=walks.device)
    probs_scaled = torch.ones(n_metapath, L, dtype=torch.double, device=walks.device)

    for pos in range(L):
        if walks[0, pos].item() != mask_token:
            continue
        with torch.inference_mode():
            output = model(walks)                               # (n_metapath, L, vocab_size)
        logits = output[:, pos, :]                              # (n_metapath, vocab_size)
        topk_logits, topk_indices = torch.topk(logits, k=5, dim=-1)
        p    = F.softmax(topk_logits,     dim=-1)
        p_sc = F.softmax(topk_logits / T, dim=-1)
        sampled = torch.multinomial(p_sc, num_samples=1).squeeze(-1)   # (n_metapath,)
        walks[:, pos]        = topk_indices.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
        probs[:, pos]        = p.gather(   1, sampled.unsqueeze(-1)).squeeze(-1).to(torch.double)
        probs_scaled[:, pos] = p_sc.gather(1, sampled.unsqueeze(-1)).squeeze(-1).to(torch.double)

    return walks, probs_scaled, probs


def apply_frequency_consensus(triplet_df, n_metapath, walk_length, threshold=0.15):
    """
    Keep only triplets that appear in >= threshold fraction of paths,
    computed separately for forward and backward paths.
    """
    triplets_per_path = walk_length - 1
    n_per_direction   = n_metapath * triplets_per_path

    forward_df  = triplet_df.iloc[:n_per_direction].copy()
    backward_df = triplet_df.iloc[n_per_direction:].copy()

    def freq_filter(df, n_paths, thr):
        freq = (
            df.groupby(['head_index', 'relation_index', 'tail_index'])
            .size()
            .reset_index(name='count')
        )
        freq['frequency'] = freq['count'] / n_paths
        df = df.merge(
            freq[['head_index', 'relation_index', 'tail_index', 'frequency']],
            on=['head_index', 'relation_index', 'tail_index'],
            how='left',
        )
        return (
            df.drop_duplicates(subset=['head_index', 'relation_index', 'tail_index'])
              .query('frequency >= @thr')
              .reset_index(drop=True)
        )

    fwd = freq_filter(forward_df,  n_metapath, threshold)
    bwd = freq_filter(backward_df, n_metapath, threshold)
    return (
        pd.concat([fwd, bwd], ignore_index=True)
          .drop_duplicates(subset=['head_index', 'relation_index', 'tail_index'])
          .reset_index(drop=True)
    )


##############################################################################################################
# Step 2: per-head subnetwork generation with LP filtering and consensus
start_total = time.time()
all_saved   = []

for i in tqdm(range(len(head_idx)), desc='Heads'):
    start      = time.time()
    head_index = head_idx[i]
    all_triplets = []

    # Batch LP scores for all top tails in a single forward pass
    top_tails_i = top_tails_all[i].to(device)
    batch_sen   = torch.full((N_actual, seq_len), mask_token_id, dtype=torch.long, device=device)
    batch_sen[:, 0] = head_index
    batch_sen[:, 1] = relation_token_id
    batch_sen[:, 2] = top_tails_i

    with torch.inference_mode():
        embs     = model.get_embeddings(batch_sen, select='transformer')
        probs_lp = F.sigmoid(classifier(embs)).squeeze(-1)

    passing_js = (probs_lp >= LP_THRESHOLD).nonzero(as_tuple=True)[0].tolist()

    if not passing_js:
        print(f"Head {head_index}: no tails passed LP threshold — skipping")
        continue

    for j in tqdm(passing_js, desc=f'  Head {head_index} (LP-passing tails)', leave=False):
        tail_index  = top_tails_i[j].item()
        prob_lp     = probs_lp[j].item()
        is_existing = f'{head_index}|{tail_index}' in relation_edges

        walk1 = torch.tensor(
            [[head_index, relation_token_id, tail_index] + [mask_token_id] * n_mask],
            device=device,
        )
        walk2 = torch.tensor(
            [[tail_index, relation_token_id, head_index] + [mask_token_id] * n_mask],
            device=device,
        )

        walks1, psc1, p1 = metapath_generation_batched(walk1, N_METAPATH, mask_token_id, model, T=TEMPERATURE)
        walks2, psc2, p2 = metapath_generation_batched(walk2, N_METAPATH, mask_token_id, model, T=TEMPERATURE)

        metapath     = torch.cat([walks1, walks2], dim=0).detach().cpu().numpy()
        probs_arr    = torch.cat([p1,     p2    ], dim=0).detach().cpu().numpy()
        probs_sc_arr = torch.cat([psc1,   psc2  ], dim=0).detach().cpu().numpy()

        triplets, prob_trips, prob_sc_trips = [], [], []
        for row, p_sc, p in zip(metapath, probs_sc_arr, probs_arr):
            for k in range(0, len(row) - 2, 2):
                triplets.append(row[k:k + 3].tolist())
                prob_trips.append(p[k:k + 3].tolist())
                prob_sc_trips.append(p_sc[k:k + 3].tolist())

        triplet_df = pd.concat([
            pd.DataFrame(triplets,      columns=['head_index', 'relation_index', 'tail_index']),
            pd.DataFrame(prob_trips,    columns=['head_prob', 'relation_prob', 'tail_prob']),
            pd.DataFrame(prob_sc_trips, columns=['head_prob_scaled', 'relation_prob_scaled', 'tail_prob_scaled']),
        ], axis=1)

        triplet_df = apply_frequency_consensus(triplet_df, N_METAPATH, param['walk_length'], FREQ_THRESHOLD)

        if triplet_df.empty:
            continue

        triplet_df['head_type']    = triplet_df['head_index'].map(index_type_map)
        triplet_df['relation_type'] = triplet_df['relation_index'].map(index_type_map)
        triplet_df['tail_type']    = triplet_df['tail_index'].map(index_type_map)
        triplet_df['head_name']    = triplet_df['head_index'].map(index_name_map)
        triplet_df['tail_name']    = triplet_df['tail_index'].map(index_name_map)
        triplet_df['pair_id']      = f'{head_index}|{tail_index}'
        triplet_df['drug_disease_score_indication'] = prob_lp
        triplet_df['drug_disease_in_KG'] = is_existing

        all_triplets.append(triplet_df)

    if not all_triplets:
        continue

    final_df = pd.concat(all_triplets, ignore_index=True)
    cols = [c for c in final_df.columns if c != 'frequency'] + ['frequency']
    final_df = final_df[cols]
    mask_cols = [c for c in final_df.columns if c != 'frequency']
    final_df[mask_cols] = final_df[mask_cols].mask(
        final_df[mask_cols].map(
            lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and x == 1
        ),
        pd.NA,
    )

    out_path = os.path.join(output_dir, f'subnetwork_{head_index}.csv')
    final_df.to_csv(out_path, index=False)
    all_saved.append(out_path)

    elapsed = (time.time() - start) / 60
    print(f"Head {head_index}: {len(final_df)} triplets in {elapsed:.3f} min — {out_path}")

total_elapsed = (time.time() - start_total) / 60
print(f"\nDone in {total_elapsed:.2f} min — {len(all_saved)} file(s) written to {output_dir}")
