"""
DREAMwalk with its own (native) walk generation, restricted to the NetMedGPT training edges.

The full DREAMwalk pipeline is run: edge-type transition matrix + teleport-guided random walks
(DREAMwalk/generate_embeddings.py, save_embedding_files) -> heterogeneous skip-gram -> XGBoost.
Only training-split edges are written to the DREAMwalk network file, so val/test edges are never seen.
Evaluation uses the same val/test pairs and metrics as NetMedGPT.

Teleport needs a drug/disease similarity network built from hierarchies (DREAMwalk generate_similarity_net.py).
  --hierarchy_file given : the similarity network is built from it and teleport is used (--tp_factor)
  --hierarchy_file empty : no similarity network, tp_factor = 0 (DREAMwalk walks without teleport)
Hierarchy file format (DREAMwalk): csv with columns child,parent; all leaves must reach a root named "drug" or "disease".
The leaf names must be KG node_index values (as strings), the same node names used in the network file.

Walk settings default to the NetMedGPT corpus settings in data/parameters.json
(walks_per_node -> num_walks, walk_length -> walk_length), so both methods sample the same amount of walks.

Method name in the results: dreamwalk_native

Example (from the repository root):
    python dreamwalk_native.py --inference random_link_split --seed 1 --gpu 0 --dreamwalk_dir /path/to/DREAMwalk \
        --hierarchy_file data/new_analysis/hierarchy.csv

Requires: a clone of https://github.com/eugenebang/DREAMwalk, g++, xgboost, parmap, networkx.
DREAMwalk keeps the whole graph in networkx and samples walks in Python, so on the full KG this needs a lot of
memory and time. Do not run two DREAMwalk jobs at the same time (fixed temporary file names inside its folder).
"""

import os
import sys
import argparse
import pickle
import subprocess
import numpy as np
import pandas as pd

from common import (load_split, build_corpus, get_device, embedding_matrix, coverage_report,
                    PairScorer, score_pairs)
from baseline_utils import (finish, work_dir, write_node_type_file, add_dreamwalk_to_path, in_dir,
                            fit_xgb_per_relation)


####### arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID")
parser.add_argument("--inference", required=False, default="random_link_split", help="random_link_split, zero_shot_split, rare_disease or a disease area")
parser.add_argument("--seed", required=False, default=1, type=int, help="Random seed / split number")
parser.add_argument("--dreamwalk_dir", required=True, help="path to the cloned DREAMwalk repository")
parser.add_argument("--hierarchy_file", required=False, default="", help="child,parent csv for the similarity network (optional)")
parser.add_argument("--cut_off", required=False, default=0.5, type=float, help="similarity cut-off (DREAMwalk default 0.5)")
parser.add_argument("--tp_factor", required=False, default=0.5, type=float, help="teleport probability (DREAMwalk default 0.5)")
parser.add_argument("--num_walks", required=False, default=None, type=int, help="walks per node; default = walks_per_node in parameters.json")
parser.add_argument("--walk_length", required=False, default=None, type=int, help="nodes per walk; default = walk_length in parameters.json")
parser.add_argument("--em_max_iter", required=False, default=5, type=int, help="EM iterations for the edge-type transition matrix (DREAMwalk default 5)")
parser.add_argument("--dim", required=False, default=300, type=int, help="embedding size (300 = NetMedGPT hidden size)")
parser.add_argument("--window", required=False, default=4, type=int, help="skip-gram window (DREAMwalk default 4)")
parser.add_argument("--workers", required=False, default=30, type=int)


# DREAMwalk network file from the training split only: tab-separated, no header,
# columns source, target, edge type (int, 1..T), weight, edge id (format read by DREAMwalk utils.read_graph).
# DREAMwalk builds an undirected graph, so each (node pair, relation) is written once.
def write_network_file(split, path):
    train = split['train_txgnn'][['x_index', 'y_index', 'relation']].copy()
    train['a'] = train[['x_index', 'y_index']].min(axis=1)
    train['b'] = train[['x_index', 'y_index']].max(axis=1)
    train = train.drop_duplicates(subset=['a', 'b', 'relation'])

    rel2type = {r: i + 1 for i, r in enumerate(sorted(train['relation'].unique()))}
    net = pd.DataFrame({
        'source': train['a'].astype(str),
        'target': train['b'].astype(str),
        'type': train['relation'].map(rel2type),
        'weight': 1.0,
        'id': np.arange(len(train)),
    })
    net.to_csv(path, sep='\t', header=False, index=False)
    print(f"Network file: {len(net)} training edges, {len(rel2type)} edge types -> {path}")
    return rel2type


# DREAMwalk similarity network from the hierarchy file (DREAMwalk generate_similarity_net.py, run as in its README)
def build_similarity_network(dreamwalk_dir, netf, hierarchy_file, simf, cut_off):
    with in_dir(dreamwalk_dir):
        subprocess.run([sys.executable, "-m", "DREAMwalk.generate_similarity_net",
                        "--network_file", netf,
                        "--hierarchy_file", hierarchy_file,
                        "--output_file", simf,
                        "--cut_off", str(cut_off)], check=True)
    return simf


def main():
    args = parser.parse_args()
    device = get_device(args.gpu)
    method = "dreamwalk_native"

    # 1. same split and same val/test pairs as NetMedGPT (the NetMedGPT walks are not used here)
    split = load_split(args.inference, args.seed)
    data = build_corpus(split)
    out_dir = work_dir(split, method)
    node2vec_param = split['all_param']['node2vec']
    num_walks = args.num_walks or node2vec_param['walks_per_node']
    walk_length = args.walk_length or node2vec_param['walk_length']

    # 2. DREAMwalk input files (training edges only)
    netf = os.path.join(out_dir, "network.txt")
    write_network_file(split, netf)
    nodetypef = write_node_type_file(split, os.path.join(out_dir, "nodetypes.tsv"))

    dreamwalk_dir = add_dreamwalk_to_path(args.dreamwalk_dir)
    if args.hierarchy_file:
        simf = build_similarity_network(dreamwalk_dir, netf, os.path.abspath(args.hierarchy_file),
                                        os.path.join(out_dir, "sim_network.txt"), args.cut_off)
        tp_factor = args.tp_factor
    else:
        print("No hierarchy file: DREAMwalk runs without teleport (tp_factor = 0)")
        simf, tp_factor = "", 0.0

    # 3. DREAMwalk walks + heterogeneous skip-gram (official save_embedding_files)
    from DREAMwalk.generate_embeddings import save_embedding_files
    embf = os.path.join(out_dir, "embedding_file.pkl")
    with in_dir(dreamwalk_dir):
        save_embedding_files(netf=netf, sim_netf=simf, outputf=embf, nodetypef=nodetypef,
                             tp_factor=tp_factor, seed=args.seed, directed=False, weighted=True,
                             em_max_iter=args.em_max_iter, num_walks=num_walks, walk_length=walk_length,
                             workers=args.workers, dimension=args.dim, window_size=args.window, p=1, q=1)
    with open(embf, "rb") as f:
        embeddings = pickle.load(f)

    E, missing = embedding_matrix(embeddings, split['vocab_size'], args.dim)
    coverage_report(missing, data)

    # 4. DREAMwalk XGBoost classifier per relation, then scoring of the val/test pairs
    pair_score_fn = fit_xgb_per_relation(split, E, args.seed, n_jobs=args.workers)
    scorer = PairScorer(pair_score_fn, split['vocab_size'], split['all_ids_node2'], device)
    pred_val = score_pairs(data['val_data_edge_label_index_with_relation'], pair_score_fn)
    pred_test = score_pairs(data['test_data_edge_label_index_with_relation'], pair_score_fn)

    # 5. same metrics and output files as NetMedGPT
    finish(split, data, pred_val, pred_test, scorer, device, method)


if __name__ == "__main__":
    main()
