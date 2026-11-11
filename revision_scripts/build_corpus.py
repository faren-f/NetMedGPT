"""
Builds and caches the walk corpus and the val/test pairs for one split and seed
(same sentence_generation call as train.py), so every baseline reads exactly the same data.

Optional: --check_reproducible 1 generates the corpus twice and checks that both are identical.
If they are, the cached corpus is the one the existing NetMedGPT checkpoints were trained on.
(This runs sentence_generation two extra times, so do it once, e.g. for random_link_split seed 1.)

Example (from the repository root):
    python build_corpus.py --inference random_link_split --seed 1
    python build_corpus.py --inference random_link_split --seed 1 --check_reproducible 1
"""

import argparse
from common import load_split, build_corpus, corpus_is_reproducible

parser = argparse.ArgumentParser()
parser.add_argument("--inference", required=False, default="random_link_split", help="random_link_split, zero_shot_split, rare_disease or a disease area")
parser.add_argument("--seed", required=False, default=1, type=int, help="Random seed / split number")
parser.add_argument("--check_reproducible", required=False, default=0, type=int, help="1: check that the corpus is identical when generated twice")
parser.add_argument("--no_cache", required=False, default=0, type=int, help="1: regenerate even if a cached corpus exists")
args = parser.parse_args()

split = load_split(args.inference, args.seed)

if args.check_reproducible:
    ok = corpus_is_reproducible(split)
    print(f"\nCorpus reproducible: {ok}")

data = build_corpus(split, use_cache=not args.no_cache)

print(f"\nwalks          : {tuple(data['node2vec_walks'].shape)}")
print(f"val pairs      : {data['val_data_edge_label_index_with_relation'].shape[1]} "
      f"({int(data['val_data_edge_label'].sum())} positive)")
print(f"test pairs     : {data['test_data_edge_label_index_with_relation'].shape[1]} "
      f"({int(data['test_data_edge_label'].sum())} positive)")
