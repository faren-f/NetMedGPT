"""
Temperature and MC-dropout uncertainty analysis of the subnetwork of one drug-disease pair.

Uses the same model, KG and path generation as generate_subnetwork.py (full-KG model data/model_checkpoints/netmedgpt.pt;
paths [head, relation, tail, MASK, ...] and [tail, relation, head, MASK, ...]; each masked position sampled from the
top-5 tokens with temperature T; N paths per direction; an edge is in the subnetwork if it occurs in >= freq_threshold
of the paths of one direction). generate_subnetwork.py itself is not changed.

Part A - temperature sweep
  For every temperature T and n_repeats independent sampling runs (different random seeds):
    subnetwork size (edges, nodes), node-type composition (fraction of molecular nodes: gene/protein, pathway,
    biological_process, molecular_function, cellular_component; fraction of effect/phenotype nodes), fraction of
    edges that already exist in the KG, mean entropy of the sampling distribution, and the stability of the
    subnetwork between runs (mean pairwise Jaccard index of the edge sets).
  Suggested temperature: the highest T whose stability is >= --stability_target (more diverse paths while the
  subnetwork stays reproducible).

Part B - MC dropout at one temperature (--mc_temperature)
  Dropout of the transformer layers (p = 0.1, as in training) is kept on at inference; every MC sample is a slightly
  different model (Gal & Ghahramani, 2016).
    1. edge frequency per MC sample (N paths each) -> mean, sd and 95% interval of the frequency, presence rate
    (reported for every edge with frequency >= --report_freq, with or without dropout)
    2. uncertainty of the model prediction that generated the edge (full softmax at that position, context = the
       path up to that position), over the MC samples p_1..p_M:
         total      H(mean_m p_m)
         aleatoric  mean_m H(p_m)                (ambiguity: several tokens plausible)
         epistemic  total - aleatoric             (mutual information: disagreement between MC samples)
       averaged over (up to --max_occurrences) occurrences of the edge in the paths.

The query triplet itself (head, relation, tail) is part of every path and is excluded from the statistics.

Outputs: data/user_response/subnetwork_uncertainty/<head>_<tail>/
  temperature_runs.csv, temperature_summary.csv, edges_T<T>.csv (edge frequencies per temperature),
  mc_dropout_edges_T<T>.csv (one per MC dropout temperature), temperature_sweep.pdf, mc_dropout_edges.pdf

Example (from the repository root, in the netmedgpt environment):
    python subnetwork_uncertainty.py --gpu 0 --head_index 25681 --tail_index 15320
    (25681 = osteogenesis imperfecta, 15320 = Bosentan in nodes_preprocessed.csv)
"""

import os
import json
import itertools
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from netmedgpt.model import TransformerModel
from config import set_seed


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID")
parser.add_argument("--head_index", required=False, default=25681, type=int, help="node_index of the head (disease)")
parser.add_argument("--tail_index", required=False, default=15320, type=int, help="node_index of the tail (drug)")
parser.add_argument("--relation_type", required=False, default="indication")
parser.add_argument("--temperatures", required=False, default="0.5,1,2,5,10")
parser.add_argument("--n_repeats", required=False, default=5, type=int, help="independent sampling runs per temperature")
parser.add_argument("--N_metapath", required=False, default=200, type=int, help="paths per direction (as generate_subnetwork.py)")
parser.add_argument("--top_k", required=False, default=5, type=int, help="tokens sampled from per position (as generate_subnetwork.py)")
parser.add_argument("--freq_threshold", required=False, default=0.15, type=float, help="as generate_subnetwork.py")
parser.add_argument("--stability_target", required=False, default=0.8, type=float, help="minimum mean Jaccard for the suggested T")
parser.add_argument("--mc_temperature", required=False, default=1.0, type=float, help="temperature of the MC dropout part")
parser.add_argument("--mc_temperatures", required=False, default="", help="several MC dropout temperatures, e.g. 0.5,1,2,5,10 (overrides --mc_temperature)")
parser.add_argument("--mc_samples", required=False, default=20, type=int, help="MC dropout samples")
parser.add_argument("--report_freq", required=False, default=0.02, type=float, help="MC dropout statistics for all edges with at least this frequency")
parser.add_argument("--max_occurrences", required=False, default=30, type=int, help="occurrences per edge for the entropy decomposition")
parser.add_argument("--out_dir", required=False, default="data/user_response/subnetwork_uncertainty")

MOLECULAR = {'gene/protein', 'pathway', 'biological_process', 'molecular_function', 'cellular_component'}


#######################################################################################################################
# Model and KG (as generate_subnetwork.py)

def load(device):
    with open("data/parameters.json") as f:
        all_param = json.load(f)
    data_dir = all_param['files']['data_dir']
    nodes = pd.read_csv(os.path.join(data_dir, 'KG/nodes_preprocessed.csv'))
    edge = pd.read_csv(os.path.join(data_dir, 'KG/edges_preprocessed.csv'))
    feat = torch.load(os.path.join(data_dir, 'KG/embeddings_preprocessedKG.pt'))

    index_relations = edge[['relation', 'z_index', 'relation']].drop_duplicates()
    index_relations.columns = ['node_type', 'node_index', 'node_name']
    node_edge_indices = pd.concat([nodes, index_relations])
    mask_token = int(edge['z_index'].max()) + 1
    relation_type = list(edge['relation'].unique())
    entity = list(nodes['node_type'].unique()) + relation_type + ['mask']
    relation_index = edge.loc[edge['relation'].isin(relation_type), ['relation', 'z_index']].drop_duplicates()
    relation_mask_index = pd.concat([relation_index, pd.DataFrame([['mask', mask_token]], columns=['relation', 'z_index'])],
                                    ignore_index=True)

    ckpt = torch.load(os.path.join(data_dir, 'model_checkpoints', 'netmedgpt.pt'), map_location=device)
    param = ckpt['parameters']
    model = TransformerModel(mask_token + 1, param['hidden_channels'], param['nhead'], param['N_encoder_layers'],
                             param['walk_length'] * 2 - 1, device=device, feat=feat, nodes=nodes, entity=entity,
                             relation_mask_index=relation_mask_index, pos_emb='fixed').to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    kg = set(zip(edge['x_index'], edge['z_index'], edge['y_index']))
    names = dict(zip(node_edge_indices['node_index'], node_edge_indices['node_name']))
    types = dict(zip(node_edge_indices['node_index'], node_edge_indices['node_type']))
    return model, edge, mask_token, param['walk_length'] * 2 - 1, kg, names, types, data_dir


# dropout of the transformer layers on (MC dropout) or off; everything else stays in eval mode
def set_mc_dropout(model, on):
    model.eval()
    if on:
        model.transformer.train()


#######################################################################################################################
# Path generation (as metapath_generation_batched in generate_subnetwork.py, plus the sampling entropy)

@torch.no_grad()
def generate(model, walk, n, mask_token, T, top_k):
    walks = walk.expand(n, -1).clone()
    entropies = []
    for pos in range(walks.shape[1]):
        if walks[0, pos].item() != mask_token:
            continue
        logits = model(walks)[:, pos, :]
        topk_logits, topk_idx = torch.topk(logits, k=top_k, dim=-1)
        p = F.softmax(topk_logits / T, dim=-1)
        entropies.append(float(-(p * torch.log(p + 1e-12)).sum(-1).mean()))
        sampled = torch.multinomial(p, num_samples=1)
        walks[:, pos] = topk_idx.gather(1, sampled).squeeze(-1)
    return walks.cpu(), float(np.mean(entropies))


# triplets (head, relation, tail) at positions 0, 2, 4, ... of every path; the first one is the query triplet
def triplets(walks):
    out = []
    for p, row in enumerate(walks.tolist()):
        for k in range(2, len(row) - 2, 2):
            out.append((p, k, row[k], row[k + 1], row[k + 2]))
    return pd.DataFrame(out, columns=['path', 'pos', 'h', 'r', 't'])


# both directions, frequency of every edge per direction; subnetwork = edges with freq >= threshold in one direction
def run(model, ctx, T, seed, n=None):
    set_seed(seed)
    n = n or ctx['N']
    w1, e1 = generate(model, ctx['walk1'], n, ctx['mask'], T, ctx['top_k'])
    w2, e2 = generate(model, ctx['walk2'], n, ctx['mask'], T, ctx['top_k'])
    t1, t2 = triplets(w1), triplets(w2)
    f1 = t1.groupby(['h', 'r', 't']).size() / n  # occurrences / paths, as apply_frequency_consensus
    f2 = t2.groupby(['h', 'r', 't']).size() / n
    freq = pd.concat([f1.rename('freq_fwd'), f2.rename('freq_bwd')], axis=1).fillna(0)
    freq['freq'] = freq[['freq_fwd', 'freq_bwd']].max(axis=1)
    return freq, (w1, w2), (t1, t2), (e1 + e2) / 2


def describe(edges, ctx):
    nodes = set(h for h, _, _ in edges) | set(t for _, _, t in edges)
    nodes -= {ctx['head'], ctx['tail']}
    nt = [ctx['types'].get(x) for x in nodes]
    return {'n_edges': len(edges), 'n_nodes': len(nodes),
            'frac_molecular_nodes': np.mean([t in MOLECULAR for t in nt]) if nt else np.nan,
            'frac_phenotype_nodes': np.mean([t == 'effect/phenotype' for t in nt]) if nt else np.nan,
            'frac_edges_in_KG': np.mean([e in ctx['kg'] for e in edges]) if edges else np.nan}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else np.nan  # undefined for two empty subnetworks


def annotate(df, ctx):
    df = df.reset_index()
    df['head_name'] = df['h'].map(ctx['names'])
    df['relation'] = df['r'].map(ctx['names'])
    df['tail_name'] = df['t'].map(ctx['names'])
    df['head_type'] = df['h'].map(ctx['types'])
    df['tail_type'] = df['t'].map(ctx['types'])
    df['in_KG'] = [(h, r, t) in ctx['kg'] for h, r, t in zip(df['h'], df['r'], df['t'])]
    return df


#######################################################################################################################
# Part A: temperature sweep

def temperature_sweep(model, ctx, temps, args, out):
    set_mc_dropout(model, False)
    runs, edge_tables = [], {}
    for T in temps:
        sets, freqs = [], []
        for r in range(args.n_repeats):
            freq, _, _, ent = run(model, ctx, T, seed=1000 * r + 1)
            sub = set(freq.index[freq['freq'] >= args.freq_threshold])
            sets.append(sub)
            freqs.append(freq['freq'].rename(f'run{r}'))
            runs.append({'T': T, 'repeat': r, 'mean_sampling_entropy': ent, **describe(sub, ctx)})
        stab = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
        for row in runs[-args.n_repeats:]:
            row['stability_jaccard'] = float(np.nanmean(stab)) if not np.all(np.isnan(stab)) else np.nan
        f = pd.concat(freqs, axis=1).fillna(0)
        f['freq_mean'], f['freq_sd'] = f.mean(1), f.std(1)
        f['in_subnetwork_runs'] = (f[[c for c in f.columns if c.startswith('run')]] >= args.freq_threshold).sum(1)
        edge_tables[T] = annotate(f.sort_values('freq_mean', ascending=False), ctx)
        edge_tables[T].to_csv(os.path.join(out, f'edges_T{T:g}.csv'), index=False)
        print(f"T={T:g}: {np.mean([len(s) for s in sets]):.1f} edges, stability (Jaccard) {runs[-1]['stability_jaccard']:.3f}")

    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(out, 'temperature_runs.csv'), index=False)
    summary = runs.groupby('T').agg(['mean', 'std']).drop(columns='repeat', level=0)
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.drop(columns='stability_jaccard_std').rename(columns={'stability_jaccard_mean': 'stability_jaccard'})
    # only temperatures that give a non-empty subnetwork
    valid = summary[summary['n_edges_mean'] > 0]
    ok = valid.index[valid['stability_jaccard'] >= args.stability_target]
    best = ok.max() if len(ok) else (valid['stability_jaccard'].idxmax() if len(valid) else None)
    summary['suggested'] = summary.index == best
    summary.to_csv(os.path.join(out, 'temperature_summary.csv'))
    print(summary.round(3).to_string())
    return summary


#######################################################################################################################
# Part B: MC dropout

def entropy(p):
    return -(p * torch.log(p + 1e-12)).sum(-1)


@torch.no_grad()
def mc_dropout(model, ctx, args, out, T=None):
    T, M = (args.mc_temperature if T is None else T), args.mc_samples

    # 1. edge frequencies per MC sample
    set_mc_dropout(model, True)
    freqs = [run(model, ctx, T, seed=500 + m)[0]['freq'].rename(f'mc{m}') for m in range(M)]
    f = pd.concat(freqs, axis=1).fillna(0)
    mc_cols = list(f.columns)
    res = pd.DataFrame({'freq_mean': f.mean(1), 'freq_sd': f.std(1),
                        'freq_q025': f.quantile(0.025, axis=1), 'freq_q975': f.quantile(0.975, axis=1),
                        'presence_rate': (f[mc_cols] >= args.freq_threshold).mean(1)})

    # deterministic reference run (dropout off): edges of the subnetwork and their occurrences in the paths
    set_mc_dropout(model, False)
    ref, (w1, w2), (t1, t2), _ = run(model, ctx, T, seed=1)
    res = res.join(ref['freq'].rename('freq_no_dropout'), how='outer').fillna(0)
    keep = res.index[(res['freq_no_dropout'] >= args.report_freq) | (res['freq_mean'] >= args.report_freq)]
    res = res.loc[keep]

    # 2. entropy decomposition of the prediction that generated each edge (tail token at position pos + 2)
    contexts, owners = [], []
    for walks, t in [(w1, t1), (w2, t2)]:
        occ = t.set_index(['h', 'r', 't']).sort_index()
        for e in res.index:
            if e not in occ.index:
                continue
            rows = occ.loc[[e]].head(args.max_occurrences)
            for path, pos in zip(rows['path'], rows['pos']):
                c = walks[path].clone()
                c[pos + 2:] = ctx['mask']  # context at the time the tail token was generated
                contexts.append((c, pos + 2, e[2]))
                owners.append(e)
    if contexts:
        X = torch.stack([c for c, _, _ in contexts]).to(ctx['device'])
        P = torch.tensor([p for _, p, _ in contexts], device=ctx['device'])
        tok = torch.tensor([t for _, _, t in contexts], device=ctx['device'])
        set_mc_dropout(model, True)
        probs = []
        for m in range(M):
            set_seed(900 + m)
            logits = torch.cat([model(X[i:i + 256])[torch.arange(len(X[i:i + 256])), P[i:i + 256], :]
                                for i in range(0, len(X), 256)])
            probs.append(F.softmax(logits, dim=-1))
        set_mc_dropout(model, False)
        probs = torch.stack(probs)                         # (M, n_contexts, vocab)
        mean_p = probs.mean(0)
        dec = pd.DataFrame({
            'edge': owners,
            'total_entropy': entropy(mean_p).cpu().numpy(),
            'aleatoric': entropy(probs).mean(0).cpu().numpy(),
            'p_tail_mean': mean_p[torch.arange(len(tok)), tok].cpu().numpy(),
            'p_tail_sd': probs[:, torch.arange(len(tok)), tok].std(0).cpu().numpy(),
        })
        dec['epistemic'] = dec['total_entropy'] - dec['aleatoric']
        dec = dec.groupby('edge').mean()
        dec.index = pd.MultiIndex.from_tuples(dec.index, names=['h', 'r', 't'])
        res = res.join(dec, how='left')

    res = annotate(res.sort_values('freq_mean', ascending=False), ctx)
    res['in_subnetwork_no_dropout'] = res['freq_no_dropout'] >= args.freq_threshold
    res.to_csv(os.path.join(out, f'mc_dropout_edges_T{T:g}.csv'), index=False)
    print(f"MC dropout (T={T:g}, {M} samples): {len(res)} edges with frequency >= {args.report_freq}")
    if len(res) and 'epistemic' in res:
        print(res[['head_name', 'relation', 'tail_name', 'tail_type', 'freq_mean', 'freq_sd', 'presence_rate',
               'aleatoric', 'epistemic', 'in_KG']].head(30).round(3).to_string(index=False))
    return res


#######################################################################################################################
# Plots

def plots(summary, mc, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T = summary.index.values
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
    for a, (col, label) in zip(ax, [('n_edges', 'edges in subnetwork'), ('frac_molecular_nodes', 'fraction molecular nodes'),
                                    ('stability_jaccard', 'stability (Jaccard)'), ('mean_sampling_entropy', 'sampling entropy')]):
        y = summary[f'{col}_mean'] if f'{col}_mean' in summary else summary[col]
        err = summary[f'{col}_std'] if f'{col}_std' in summary else None
        a.errorbar(T, y, yerr=err, marker='o')
        a.set_xscale('log')
        a.set_xlabel('temperature T')
        a.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'temperature_sweep.pdf'))
    plt.close(fig)

    if 'epistemic' in mc:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for typ, g in mc.groupby(mc['tail_type'].map(lambda t: 'molecular' if t in MOLECULAR else t)):
            ax.scatter(g['freq_mean'], g['epistemic'], label=typ, s=25)
        ax.set_xlabel('edge frequency (mean over MC dropout samples)')
        ax.set_ylabel('epistemic uncertainty (mutual information)')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'mc_dropout_edges.pdf'))
        plt.close(fig)


def main():
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model, edge, mask_token, seq_len, kg, names, types, data_dir = load(device)
    rel = int(edge.loc[edge['relation'] == args.relation_type, 'z_index'].unique().item())
    h, t = args.head_index, args.tail_index
    print(f"pair: {names.get(h)} ({types.get(h)}) - {args.relation_type} - {names.get(t)} ({types.get(t)}) | "
          f"edge in KG: {(h, rel, t) in kg or (t, rel, h) in kg}")

    n_mask = seq_len - 3
    ctx = {'walk1': torch.tensor([[h, rel, t] + [mask_token] * n_mask], device=device),
           'walk2': torch.tensor([[t, rel, h] + [mask_token] * n_mask], device=device),
           'mask': mask_token, 'N': args.N_metapath, 'top_k': args.top_k, 'kg': kg, 'names': names, 'types': types,
           'head': h, 'tail': t, 'device': device}
    out = os.path.join(args.out_dir, f"{h}_{t}")
    os.makedirs(out, exist_ok=True)

    summary = temperature_sweep(model, ctx, [float(x) for x in args.temperatures.split(',')], args, out)
    mc_temps = [float(x) for x in args.mc_temperatures.split(',')] if args.mc_temperatures else [args.mc_temperature]
    mc = {T: mc_dropout(model, ctx, args, out, T) for T in mc_temps}
    plots(summary, mc[mc_temps[0]], out)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
