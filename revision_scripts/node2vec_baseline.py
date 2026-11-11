"""
node2vec baselines trained on the identical walk corpus used by NetMedGPT.

Corpus variants (--keep_relations)
  0 : classic node2vec, skip-gram on node-only walks         [n0, n1, n2, n3, n4]
  1 : skip-gram on the relation-augmented sentences NetMedGPT sees [n0, r01, n1, r12, n2, r23, n3, r34, n4]

Scoring (--scoring)
  cosine     : zero-shot, cosine(head, tail) through the same sim() / node_level_eval() as NetMedGPT
               (relation-agnostic: the same pair gets the same score for every relation)
  classifier : one logistic regression per relation on the Hadamard product of head and tail embeddings,
               trained on training-split pairs (negatives sampled like the val/test negatives)

Method names in the results: node2vec_cosine, node2vec_classifier, node2vec_rel_cosine, node2vec_rel_classifier

Example (from the repository root):
    python node2vec_baseline.py --inference random_link_split --seed 1 --gpu 0 --keep_relations 0 --scoring cosine

Requires gensim (not part of the netmedgpt environment).
"""

import argparse
import torch
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

from common import (load_split, build_corpus, get_device, walks_to_sentences, embedding_matrix, coverage_report,
                    EmbeddingScorer, PairScorer, predict_with_sim, score_pairs, training_pairs)
from baseline_utils import relation_tokens, finish


####### arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID")
parser.add_argument("--inference", required=False, default="random_link_split", help="random_link_split, zero_shot_split, rare_disease or a disease area")
parser.add_argument("--seed", required=False, default=1, type=int, help="Random seed / split number")
parser.add_argument("--keep_relations", required=False, default=0, type=int, help="0: node-only walks, 1: keep relation tokens")
parser.add_argument("--scoring", required=False, default="cosine", choices=["cosine", "classifier"])
# skip-gram hyperparameters; dim 300 = NetMedGPT hidden size, window 4 = context of a 5-node walk
parser.add_argument("--dim", required=False, default=300, type=int)
parser.add_argument("--window", required=False, default=4, type=int)
parser.add_argument("--negative", required=False, default=5, type=int, help="negative samples per positive in skip-gram")
parser.add_argument("--epochs", required=False, default=5, type=int)
parser.add_argument("--workers", required=False, default=30, type=int)


#######################################################################################################################
# logistic regression per relation on Hadamard(head, tail) features.
# The trained weights are applied with torch on the GPU, which keeps Recall@100 (scoring all candidates) fast.
def fit_logreg_per_relation(split, E, device, seed):
    E_dev = E.to(device)
    weights = {}
    for rel, z in relation_tokens(split).items():
        heads, tails, labels = training_pairs(split, rel)
        X = (E[heads] * E[tails]).numpy()
        clf = LogisticRegression(max_iter=1000, random_state=seed).fit(X, labels.numpy())
        weights[z] = (torch.tensor(clf.coef_[0], dtype=torch.float32, device=device), float(clf.intercept_[0]))
        print(f"Logistic regression trained for {rel}: {int(labels.sum())} positives, {int((labels == 0).sum())} negatives")

    def pair_score_fn(heads, tails, rel):
        w, b = weights[rel]
        x = E_dev[heads.to(device)] * E_dev[tails.to(device)]
        return torch.sigmoid(x @ w + b)

    return pair_score_fn


#######################################################################################################################
def main():
    args = parser.parse_args()
    device = get_device(args.gpu)
    method = f"node2vec{'_rel' if args.keep_relations else ''}_{args.scoring}"
    print(f"Method: {method}")

    # 1. same split, same walk corpus and same val/test pairs as NetMedGPT
    split = load_split(args.inference, args.seed)
    data = build_corpus(split)

    # 2. skip-gram on the walk corpus (no new walks are sampled)
    sentences = walks_to_sentences(data['node2vec_walks'], keep_relations=bool(args.keep_relations))
    w2v = Word2Vec(sentences, vector_size=args.dim, window=args.window, sg=1, negative=args.negative,
                   min_count=1, epochs=args.epochs, workers=args.workers, seed=args.seed)
    E, missing = embedding_matrix(w2v.wv, split['vocab_size'], args.dim)
    coverage_report(missing, data)

    # 3. scoring of the val/test pairs
    if args.scoring == "cosine":
        scorer = EmbeddingScorer(E, device)
        pred_val, pred_test = predict_with_sim(scorer, data, split, device)
    else:
        pair_score_fn = fit_logreg_per_relation(split, E, device, args.seed)
        scorer = PairScorer(pair_score_fn, split['vocab_size'], split['all_ids_node2'], device)
        pred_val = score_pairs(data['val_data_edge_label_index_with_relation'], pair_score_fn)
        pred_test = score_pairs(data['test_data_edge_label_index_with_relation'], pair_score_fn)

    # 4. same metrics and output files as NetMedGPT
    finish(split, data, pred_val, pred_test, scorer, device, method)


if __name__ == "__main__":
    main()
