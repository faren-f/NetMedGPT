"""
Run all seeds:

cd /home/bbc8731/NetMedGPT && for s in 1 2 3 4 5; do /home/bbc8731/miniconda3/envs/walk_baselines/bin/python revision_scripts/memlimit.py --limit_gb 250 --log dreamwalk_full_memlimit.log -- /home/bbc8731/miniconda3/envs/walk_baselines/bin/python revision_scripts/dreamwalk_full.py --gpu 0 --seed $s --workers 20 --dreamwalk_dir /home/bbc8731/DREAMwalk 2>&1 | tee dreamwalk_full_seed$s.log; done

rare disaeses:
cd /home/bbc8731/NetMedGPT && for s in 1 2 3 4 5; do /home/bbc8731/miniconda3/envs/walk_baselines/bin/python revision_scripts/memlimit.py --limit_gb 250 --log dreamwalk_full_rare_memlimit.log -- /home/bbc8731/miniconda3/envs/walk_baselines/bin/python revision_scripts/dreamwalk_full.py --inference rare_disease --gpu 0 --seed $s --workers 25 --dreamwalk_dir /home/bbc8731/DREAMwalk 2>&1 | tee dreamwalk_full_rare_seed$s.log; done

DREAMwalk (Bang et al., Nat Commun 2023) on the NetMedGPT training graph, evaluated on the NetMedGPT val/test pairs.

The whole DREAMwalk pipeline is run with the settings reported in the paper / its code defaults:
  1. semantic similarity network from the drug (ATC, DrugBank) and disease (MeSH) hierarchies,
     DREAMwalk's information-content / Jiang-Conrath similarity, cut-off 0.4 (paper)
  2. teleport-guided random walks: edge-type transition matrix (5 EM iterations), teleport factor 0.3 (paper),
     100 walks per node, walk length 10 (code defaults)
  3. heterogeneous skip-gram, 128 dimensions, window 4 (code defaults)
  4. XGBoost on drug - disease embedding differences (DREAMwalk hyperparameters), one classifier per relation
Only training-split edges are written to the DREAMwalk network file, so val/test edges are never seen.

Deviations from the original code (do not change DREAMwalk's results where it runs):
  - the similarity network is computed with DREAMwalk's own generate_sim_graph, with the node list passed as a set
    (identical output; avoids a list lookup that is very slow on a KG of this size);
  - guard for zero transition weights: when all edge-type transition weights of the next step are zero, DREAMwalk's
    walk step crashes (random.choices with zero weights); the step then picks a neighbouring edge uniformly instead.

The hierarchy file is built by revision_scripts/dreamwalk_hierarchy.ipynb.
Method name in the results: dreamwalk_full (same output files and long-format table as the other walk baselines).

Timing test (small, separate method name):
    python revision_scripts/dreamwalk_full.py --seed 1 --dreamwalk_dir /home/bbc8731/DREAMwalk \
        --num_walks 1 --em_max_iter 1 --method dreamwalk_full_test
    (method names ending in _test are not added to the long-format table of the statistical tests)
Do not run two DREAMwalk jobs at the same time (fixed temporary file names inside its folder).
"""

import os
import sys
import time
import random
import pickle
import argparse
import pandas as pd

sys.path.insert(0, os.getcwd())   # repository root (config.py) when run as revision_scripts/dreamwalk_full.py
# DREAMwalk compiles its skip-gram with g++: use the compiler of this Python environment (walk_baselines)
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

from common import (load_split, build_corpus, get_device, embedding_matrix, coverage_report,
                    PairScorer, score_pairs, evaluate, save_results)
from baseline_utils import (finish, work_dir, write_node_type_file, add_dreamwalk_to_path, in_dir,
                            fit_xgb_per_relation)
from dreamwalk_native import write_network_file


####### arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID (only for Recall@100 scoring)")
parser.add_argument("--inference", required=False, default="random_link_split")
parser.add_argument("--seed", required=False, default=1, type=int)
parser.add_argument("--dreamwalk_dir", required=True, help="path to the cloned DREAMwalk repository")
parser.add_argument("--hierarchy_file", required=False, default="data/dreamwalk_inputs/hierarchy_full_kg.csv")
parser.add_argument("--cut_off", required=False, default=0.4, type=float, help="similarity cut-off (paper: 0.4)")
parser.add_argument("--tp_factor", required=False, default=0.3, type=float, help="teleport factor (paper: 0.3)")
parser.add_argument("--num_walks", required=False, default=100, type=int, help="walks per node (DREAMwalk default 100)")
parser.add_argument("--walk_length", required=False, default=10, type=int, help="nodes per walk (DREAMwalk default 10)")
parser.add_argument("--em_max_iter", required=False, default=5, type=int, help="EM iterations (DREAMwalk default 5)")
parser.add_argument("--dim", required=False, default=128, type=int, help="embedding size (DREAMwalk default 128)")
parser.add_argument("--window", required=False, default=4, type=int, help="skip-gram window (DREAMwalk default 4)")
parser.add_argument("--workers", required=False, default=30, type=int)
parser.add_argument("--method", required=False, default="dreamwalk_full", help="method name in the result files")


# DREAMwalk similarity network (generate_similarity_net.save_sim_graph), node list passed as a set
def build_similarity_network(netf, hierarchy_file, simf, cut_off):
    from DREAMwalk.generate_similarity_net import generate_sim_graph
    net = pd.read_csv(netf, sep='\t', header=None, dtype=str)
    nodes = set(net[0]) | set(net[1])
    hier_df = pd.read_csv(hierarchy_file, dtype=str)
    sim_values = generate_sim_graph(hier_df, nodes, cut_off, directed=False)

    with open(simf, 'w') as fw:
        index = 0
        for type_number, ntype in enumerate(sim_values.keys(), start=1):
            for id1, id2, weight in sim_values[ntype]:
                fw.write(f"{id1}\t{id2}\t{type_number}\t{weight}\t{index}\n")
                index += 1
    print(f"Similarity network: {index} pairs ({', '.join(f'{k}: {len(v)}' for k, v in sim_values.items())}) -> {simf}")
    return simf


# guard for zero transition weights in DREAMwalk's walk step (see module docstring)
def guard_zero_transition_weights():
    import DREAMwalk.generate_embeddings as ge
    original = ge._network_traverse

    def network_traverse(cur, prev, G, trans_matrix, p, q):
        try:
            return original(cur, prev, G, trans_matrix, p, q)
        except ValueError:   # random.choices: total of weights must be greater than zero
            nbrs = [(nbr, e['type']) for nbr in sorted(G.neighbors(cur)) for e in G[cur][nbr].values()]
            return random.choice(nbrs)

    ge._network_traverse = network_traverse


def main():
    args = parser.parse_args()
    device = get_device(args.gpu)
    t0 = time.time()

    def stamp(msg):
        print(f"[{(time.time() - t0) / 60:7.1f} min] {msg}", flush=True)

    # 1. same split and same val/test pairs as NetMedGPT (the NetMedGPT walks are not used)
    split = load_split(args.inference, args.seed)
    data = build_corpus(split)
    out_dir = work_dir(split, args.method)

    # 2. DREAMwalk input files (training edges only) and similarity network
    # reverse edges of the split files (rev_<relation>) are the same KG relation: one edge type per KG relation
    train = split['train_txgnn'].copy()
    train['relation'] = train['relation'].str.replace(r'^rev_', '', regex=True)
    netf = os.path.join(out_dir, "network.txt")
    write_network_file(dict(split, train_txgnn=train), netf)
    nodetypef = write_node_type_file(split, os.path.join(out_dir, "nodetypes.tsv"))
    dreamwalk_dir = add_dreamwalk_to_path(args.dreamwalk_dir)

    simf = os.path.join(out_dir, f"sim_network_cutoff{args.cut_off}.txt")
    if os.path.exists(simf):
        stamp(f"Using existing similarity network {simf}")
    else:
        stamp("Building similarity network...")
        build_similarity_network(netf, os.path.abspath(args.hierarchy_file), simf, args.cut_off)
    stamp("Similarity network ready")

    # 3. DREAMwalk walks + heterogeneous skip-gram (official save_embedding_files)
    guard_zero_transition_weights()
    from DREAMwalk.generate_embeddings import save_embedding_files
    embf = os.path.join(out_dir, "embedding_file.pkl")
    with in_dir(dreamwalk_dir):
        save_embedding_files(netf=netf, sim_netf=simf, outputf=embf, nodetypef=nodetypef,
                             tp_factor=args.tp_factor, seed=args.seed, directed=False, weighted=True,
                             em_max_iter=args.em_max_iter, num_walks=args.num_walks, walk_length=args.walk_length,
                             workers=args.workers, dimension=args.dim, window_size=args.window, p=1, q=1)
    stamp("Embeddings ready")
    with open(embf, "rb") as f:
        embeddings = pickle.load(f)
    E, missing = embedding_matrix(embeddings, split['vocab_size'], args.dim)
    coverage_report(missing, data)

    # 4. DREAMwalk XGBoost classifier per relation, scoring of the NetMedGPT val/test pairs
    pair_score_fn = fit_xgb_per_relation(split, E, args.seed, n_jobs=args.workers)
    scorer = PairScorer(pair_score_fn, split['vocab_size'], split['all_ids_node2'], device)
    pred_val = score_pairs(data['val_data_edge_label_index_with_relation'], pair_score_fn)
    pred_test = score_pairs(data['test_data_edge_label_index_with_relation'], pair_score_fn)

    # 5. same metrics and output files as NetMedGPT and the other walk baselines
    #    (test runs are not added to the long-format table used by the statistical tests)
    if args.method.endswith("_test"):
        save_results(evaluate(pred_val, pred_test, data, split, scorer, device), split, args.method)
    else:
        finish(split, data, pred_val, pred_test, scorer, device, args.method)
    stamp("Done")


if __name__ == "__main__":
    main()
