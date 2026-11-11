"""
DREAMwalk on the identical walk corpus used by NetMedGPT.

DREAMwalk's own walk generation (teleport-guided random walk) is NOT used here. The NetMedGPT walks
(node-only, [n0, n1, n2, n3, n4]) are given to DREAMwalk's heterogeneous skip-gram, and pairs are
classified with DREAMwalk's XGBoost setup (embedding difference as feature, DREAMwalk hyperparameters).
This isolates the learner: same corpus, DREAMwalk embedding + classifier vs. NetMedGPT.

Method name in the results: dreamwalk_samecorpus

Example (from the repository root):
    python dreamwalk_samecorpus.py --inference random_link_split --seed 1 --gpu 0 --dreamwalk_dir /path/to/DREAMwalk

Requires: a clone of https://github.com/eugenebang/DREAMwalk (passed with --dreamwalk_dir), g++ (DREAMwalk compiles
its C++ skip-gram), and xgboost. Do not run two DREAMwalk jobs at the same time (DREAMwalk uses fixed temporary
file names inside its folder).
"""

import os
import argparse
import pickle

from common import (load_split, build_corpus, get_device, walks_to_sentences, embedding_matrix, coverage_report,
                    PairScorer, score_pairs)
from baseline_utils import finish, work_dir, write_node_type_file, hetsg_embeddings, fit_xgb_per_relation


####### arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID")
parser.add_argument("--inference", required=False, default="random_link_split", help="random_link_split, zero_shot_split, rare_disease or a disease area")
parser.add_argument("--seed", required=False, default=1, type=int, help="Random seed / split number")
parser.add_argument("--dreamwalk_dir", required=True, help="path to the cloned DREAMwalk repository")
parser.add_argument("--dim", required=False, default=300, type=int, help="embedding size (300 = NetMedGPT hidden size)")
parser.add_argument("--window", required=False, default=4, type=int, help="skip-gram window (DREAMwalk default 4)")
parser.add_argument("--workers", required=False, default=30, type=int)
parser.add_argument("--use_hetsg", required=False, default=1, type=int, help="1: heterogeneous skip-gram (DREAMwalk), 0: plain skip-gram")


def main():
    args = parser.parse_args()
    device = get_device(args.gpu)
    method = "dreamwalk_samecorpus"

    # 1. same split, same walk corpus and same val/test pairs as NetMedGPT
    split = load_split(args.inference, args.seed)
    data = build_corpus(split)
    out_dir = work_dir(split, method)

    # 2. DREAMwalk heterogeneous skip-gram on the NetMedGPT walks (node-only)
    sentences = walks_to_sentences(data['node2vec_walks'], keep_relations=False)
    nodetypef = write_node_type_file(split, os.path.join(out_dir, "nodetypes.tsv"))
    embeddings = hetsg_embeddings(sentences, nodetypef, args.dim, args.window, args.workers,
                                  args.dreamwalk_dir, use_hetsg=bool(args.use_hetsg))
    with open(os.path.join(out_dir, "embedding_file.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

    E, missing = embedding_matrix(embeddings, split['vocab_size'], args.dim)
    coverage_report(missing, data)

    # 3. DREAMwalk XGBoost classifier per relation, then scoring of the val/test pairs
    pair_score_fn = fit_xgb_per_relation(split, E, args.seed, n_jobs=args.workers)
    scorer = PairScorer(pair_score_fn, split['vocab_size'], split['all_ids_node2'], device)
    pred_val = score_pairs(data['val_data_edge_label_index_with_relation'], pair_score_fn)
    pred_test = score_pairs(data['test_data_edge_label_index_with_relation'], pair_score_fn)

    # 4. same metrics and output files as NetMedGPT
    finish(split, data, pred_val, pred_test, scorer, device, method)


if __name__ == "__main__":
    main()
