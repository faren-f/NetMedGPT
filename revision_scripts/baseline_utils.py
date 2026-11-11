"""
Helpers shared by the walk-embedding baseline scripts
(node2vec_baseline.py, dreamwalk_samecorpus.py, dreamwalk_native.py).

General part : relation tokens, final evaluation + saving (uses common.py).
DREAMwalk part: wrappers around the official DREAMwalk code (https://github.com/eugenebang/DREAMwalk).
                The DREAMwalk repository is not copied here; its folder is passed with --dreamwalk_dir.
"""

import os
import sys
import contextlib
import numpy as np
import pandas as pd
import torch

from common import training_pairs, evaluate, save_results, append_long_format

#######################################################################################################################
# General helpers

# relation name -> relation token (z_index) for the queried edge types of the split.
# Each queried relation has exactly one z_index (sentence_generation checks the same thing).
def relation_tokens(split):
    edge = split['edge']
    return {rel: int(edge.loc[edge['relation'] == rel, 'z_index'].unique().item())
            for rel in split['queried_edge_types']}


# evaluation and saving, identical for every baseline:
# evaluate() -> per-split result CSVs (save_results) -> long-format table for the statistics (append_long_format)
def finish(split, data, pred_val, pred_test, scorer, device, method):
    results = evaluate(pred_val, pred_test, data, split, scorer, device)
    LP_result_test_df, hit_precision_test = save_results(results, split, method)
    append_long_format(LP_result_test_df, hit_precision_test, split, method)
    return results


# working folder for intermediate files of one baseline run (node type file, network file, embeddings)
def work_dir(split, method):
    path = os.path.join(split['data_dir'], 'new_analysis', method, f"{split['inference']}_seed{split['seed']}")
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


#######################################################################################################################
# DREAMwalk helpers

# KG node type -> DREAMwalk node type. DREAMwalk's heterogeneous skip-gram knows 'drug', 'disease', 'gene';
# every other KG node type is passed as 'etc'.
DREAMWALK_TYPES = {'drug': 'drug', 'disease': 'disease', 'gene/protein': 'gene'}

# XGBoost hyperparameters as fixed in DREAMwalk predict_associations.py (predict_dda).
# The seed is passed separately as random_state.
DREAMWALK_XGB_PARAMS = dict(
    base_score=0.5, booster='gbtree', eval_metric='error', objective='binary:logistic',
    gamma=0, learning_rate=0.1, max_depth=6, n_estimators=500, tree_method='auto',
    min_child_weight=4, subsample=0.8, colsample_bytree=0.9, scale_pos_weight=1, max_delta_step=1,
)


# DREAMwalk compiles its C++ skip-gram and writes temporary files with paths relative to the current
# working directory (DREAMwalk/HeterogeneousSG.cpp, ./HetSG, tmp_walkfile, tmp_outputf.txt),
# so its functions are called from inside the DREAMwalk repository folder.
# Because of these fixed temporary file names, do not run two DREAMwalk jobs at the same time.
@contextlib.contextmanager
def in_dir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# makes `import DREAMwalk...` work; dreamwalk_dir is the cloned repository (the folder that contains DREAMwalk/)
def add_dreamwalk_to_path(dreamwalk_dir):
    dreamwalk_dir = os.path.abspath(dreamwalk_dir)
    if dreamwalk_dir not in sys.path:
        sys.path.insert(0, dreamwalk_dir)
    return dreamwalk_dir


# node type file in DREAMwalk format: tab-separated, header "node<TAB>type" (DREAMwalk skips the first line).
# Node names are the KG node_index values as strings, i.e. the same tokens used in the walks.
def write_node_type_file(split, path):
    nodes = split['nodes']
    df = pd.DataFrame({
        'node': nodes['node_index'].astype(str),
        'type': nodes['node_type'].map(DREAMWALK_TYPES).fillna('etc'),
    })
    df.to_csv(path, sep='\t', index=False)
    return path


# DREAMwalk heterogeneous skip-gram (DREAMwalk/HeterogeneousSG.py) on a given list of walks.
# Returns {node (str): embedding vector}.
def hetsg_embeddings(walk_sentences, nodetypef, dim, window, workers, dreamwalk_dir, use_hetsg=True):
    dreamwalk_dir = add_dreamwalk_to_path(dreamwalk_dir)
    from DREAMwalk.HeterogeneousSG import HeterogeneousSG

    nodetypef = os.path.abspath(nodetypef)
    nodes = set(t for s in walk_sentences for t in s)
    with in_dir(dreamwalk_dir):
        embeddings = HeterogeneousSG(use_hetsg, walk_sentences, nodes, nodetypef=nodetypef,
                                     embedding_size=dim, window_length=window, workers=workers)
    return embeddings


# DREAMwalk pair feature: difference of the two node embeddings.
# DREAMwalk uses drug - disease; our disease|drug pairs are (head=disease, tail=drug), so tail - head
# gives exactly drug - disease. For the other relations the same tail - head convention is used.
def dreamwalk_pair_features(E, heads, tails):
    return E[tails] - E[heads]


# one XGBoost classifier per queried relation (DREAMwalk hyperparameters), trained on training-split pairs
# from training_pairs() in common.py. Returns pair_score_fn(heads, tails, relation_token) -> probabilities,
# which is what score_pairs() and PairScorer in common.py expect.
# Note: the DREAMwalk script splits its own pair file into train/val/test; here the classifier is trained on
# the NetMedGPT training split and evaluated on the NetMedGPT val/test pairs, so all methods see the same pairs.
def fit_xgb_per_relation(split, E, seed, n_jobs=None):
    from xgboost import XGBClassifier

    models = {}
    for rel, z in relation_tokens(split).items():
        heads, tails, labels = training_pairs(split, rel)
        X = dreamwalk_pair_features(E, heads, tails).numpy()
        clf = XGBClassifier(**DREAMWALK_XGB_PARAMS, random_state=seed, n_jobs=n_jobs)
        clf.fit(X, labels.numpy())
        models[z] = clf
        print(f"XGBoost trained for {rel}: {int(labels.sum())} positives, {int((labels == 0).sum())} negatives")

    def pair_score_fn(heads, tails, rel):
        X = dreamwalk_pair_features(E, heads.cpu(), tails.cpu()).numpy()
        return models[rel].predict_proba(X)[:, 1]

    return pair_score_fn
