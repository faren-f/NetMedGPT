"""
Create rare disease splits for seeds 1–5.
The test set is identical across seeds (all rare disease drug edges).
Only the val 10% sampling differs per seed.
"""

import os
import pandas as pd
import numpy as np

VAL_FRAC = 0.10
DRUG_DISEASE_RELS = ["indication", "contraindication", "off-label use"]

nodes = pd.read_csv("data/KG/nodes.csv", low_memory=False)
edges = pd.read_csv("data/KG/edges.csv", low_memory=False)

idx2id   = nodes.set_index("node_index")["node_id"].to_dict()
idx2type = nodes.set_index("node_index")["node_type"].to_dict()

rare = pd.read_csv("data/orphnet_rare_diseases/orphanet_in_kg.csv")
rare_nodes = set(rare["kg_node_index"].astype(int))

drug_disease_edges = edges[edges["relation"].isin(DRUG_DISEASE_RELS)].copy()
rare_with_drugs = set(
    drug_disease_edges[drug_disease_edges["y_index"].isin(rare_nodes)]["y_index"].unique()
)

test_mask = drug_disease_edges["y_index"].isin(rare_with_drugs)
test_edges = drug_disease_edges[test_mask].copy()
train_drug_edges = drug_disease_edges[~test_mask].copy()
non_drug_edges = edges[~edges["relation"].isin(DRUG_DISEASE_RELS)].copy()

def build_split(df):
    out = pd.DataFrame()
    out["x_type"] = df["x_index"].map(idx2type)
    out["x_id"] = df["x_index"].map(idx2id)
    out["relation"] = df["relation"].values
    out["y_type"] = df["y_index"].map(idx2type)
    out["y_id"] = df["y_index"].map(idx2id)
    out["x_idx"] = df["x_index"].values.astype(float)
    out["y_idx"] = df["y_index"].values.astype(float)
    return out

test_out = build_split(test_edges)

for seed in [1, 2, 3, 4, 5]:
    print(f"\n=== Seed {seed} ===")
    np.random.seed(seed)

    val_indices = []
    for disease_node, grp in train_drug_edges.groupby("y_index"):
        n_val = max(1, round(len(grp) * VAL_FRAC))
        val_indices.extend(grp.sample(n=min(n_val, len(grp)), random_state=seed).index.tolist())

    val_edges = train_drug_edges.loc[val_indices].copy()
    train_drug_edges_final = train_drug_edges.drop(index=val_indices).copy()
    train_all = pd.concat([non_drug_edges, train_drug_edges_final], ignore_index=True)

    train_out = build_split(train_all)
    val_out = build_split(val_edges)

    out_dir = f"data/TxGNN_splits/rare_disease_{seed}"
    os.makedirs(out_dir, exist_ok=True)
    train_out.to_csv(f"{out_dir}/train.csv", index=False)
    val_out.to_csv(f"{out_dir}/valid.csv",   index=False)
    test_out.to_csv(f"{out_dir}/test.csv",   index=False)

    print(f"train: {len(train_out):,}  val: {len(val_out):,}  test: {len(test_out):,}")
    print(f"Saved to {out_dir}/")
    print("\n Test edge breakdown by relation:")
    print(test_out["relation"].value_counts().to_string())
    print("\n Val edge breakdown by relation:")
    print(val_out["relation"].value_counts().to_string())

print("\nDone.")
