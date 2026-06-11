import os
import sys
import json
import pickle
import glob
import re
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

os.chdir("/home/bbc8731/NetMedGPT")
sys.path.insert(0, "/home/bbc8731/NetMedGPT")

from sklearn.metrics import precision_recall_curve
from netmedgpt.utilities import link_pred_val, node_level_eval, sim
from netmedgpt.model import TransformerModel
from netmedgpt.prepare_txgnn_splits import prepare_txgnn_data
from netmedgpt.sentence_generation import negative_samples, to_edge_tensor, set_seed

def build_test_edges(edge, test_txgnn, queried_edge_types, queried_node_types, relation_node_types, seed):
    """Build test edge label tensors with relation tokens (no node2vec walks needed)."""
    queried_dict = {rel: relation_node_types[rel] for rel in queried_edge_types}

    test_data_edge_label_index = torch.empty((2, 0), dtype=torch.long)
    test_data_edge_label = torch.tensor([], dtype=torch.long)
    edge_type_flag_test = torch.tensor([], dtype=torch.long)

    for r, (rel, _) in enumerate(queried_dict.items()):
        set_seed(seed)
        edges_i = edge.loc[
            (edge["relation"] == rel) & (edge["node_types"] == queried_node_types[r]),
            ["x_index", "y_index", "xy"]
        ].copy().drop_duplicates()

        test_edges = test_txgnn.loc[
            test_txgnn["relation"] == rel, ["x_index", "y_index", "xy"]
        ].copy()
        all_tail_ids = edges_i["y_index"].unique()

        neg = negative_samples(test_edges, edge, all_tail_ids, seed)
        pos = list(zip(test_edges["x_index"], test_edges["y_index"]))

        pos_t = to_edge_tensor(pos)
        neg_t = to_edge_tensor(neg)

        idx_i = torch.cat((pos_t, neg_t), dim=1)
        lbl_i = torch.cat((
            torch.ones(pos_t.shape[1], dtype=torch.long),
            torch.zeros(neg_t.shape[1], dtype=torch.long),
        ))
        flag_i = (r + 1) * torch.ones(lbl_i.shape, dtype=torch.long)

        test_data_edge_label_index = torch.cat((test_data_edge_label_index, idx_i), dim=1)
        test_data_edge_label = torch.cat((test_data_edge_label, lbl_i))
        edge_type_flag_test = torch.cat((edge_type_flag_test, flag_i))

    # Append relation token row, reorder to [head, relation, tail]
    rel_row_all = torch.empty((1, 0), dtype=torch.long)
    for i, rel in enumerate(queried_edge_types):
        z_vals = edge.loc[edge["relation"] == rel, "z_index"].unique()
        if len(z_vals) != 1:
            raise ValueError(f"Expected 1 z_index for '{rel}', got {z_vals}")
        cols_i = test_data_edge_label_index[:, edge_type_flag_test == i + 1]
        rel_row_i = torch.full((1, cols_i.shape[1]), int(z_vals[0]), dtype=torch.long)
        rel_row_all = torch.cat([rel_row_all, rel_row_i], dim=1)

    label_index_with_rel = torch.cat([test_data_edge_label_index, rel_row_all], dim=0)
    label_index_with_rel = label_index_with_rel[[0, 2, 1], :]  # [head, relation, tail]
    return label_index_with_rel, test_data_edge_label, edge_type_flag_test


def parse_checkpoint(fname):
    """
    Returns (inference_label, split_dir, queried_edge_types, queried_node_types,
              edge_flags, seed, use_dropna) or None if unrecognised.
    """
    m = re.match(r"netmedgpt_(.+)_seed(\d+)$", fname)
    if not m:
        return None
    area_raw = m.group(1)
    seed = int(m.group(2))

    if area_raw == "randomLinkSplit":
        return (
            "random_link_split",
            f"random_{seed}",
            ["indication", "contraindication", "off-label use", "drug_protein", "drug_effect"],
            ["disease|drug", "disease|drug", "disease|drug", "drug|gene/protein", "drug|effect/phenotype"],
            [1, 2, 3, 4, 5],
            seed,
            False,
        )
    if area_raw == "zeroShot":
        return (
            "zero_shot_split",
            f"complex_disease_{seed}",
            ["indication", "contraindication"],
            ["disease|drug", "disease|drug"],
            [1, 2],
            seed,
            False,
        )
    # disease-area checkpoints: adrenal_gland, anemia, autoimmune, ...
    split_dir = f"{area_raw}_{seed}"
    if os.path.isdir(f"data/TxGNN_splits/{split_dir}"):
        return (
            area_raw,
            split_dir,
            ["indication", "contraindication"],
            ["disease|drug", "disease|drug"],
            [1, 2],
            seed,
            True,   # dropna needed for disease-area CSVs
        )
    return None


def main():
    print("Loading shared KG data...")
    with open("data/parameters.json", "r") as f:
        all_param = json.load(f)

    data_dir = all_param["files"]["data_dir"]
    nodes = pd.read_csv(os.path.join(data_dir, "KG/nodes.csv"), sep=",")
    edge = pd.read_csv(os.path.join(data_dir, "KG/edges.csv"))
    feat = torch.load(os.path.join(data_dir, "KG/embeddings.pt"))

    with open(f"{data_dir}/KG/relation_node_types.pkl", "rb") as f:
        relation_node_types = pickle.load(f)

    mask_token = int(edge["z_index"].max()) + 1
    vocab_size = mask_token + 1
    relation_type = list(edge["relation"].unique())
    node_types = list(nodes["node_type"].unique())
    entity = node_types + relation_type + ["mask"]
    relation_index = edge.loc[
        edge["relation"].isin(relation_type), ["relation", "z_index"]
    ].drop_duplicates()
    mask_row = pd.DataFrame([["mask", mask_token]], columns=["relation", "z_index"])
    relation_mask_index = pd.concat([relation_index, mask_row], ignore_index=True)

    all_ids = {
        "drug": torch.tensor(
            nodes.loc[nodes["node_type"] == "drug", "node_index"].unique(), dtype=torch.int64
        ),
        "gene": torch.tensor(
            nodes.loc[nodes["node_type"] == "gene/protein", "node_index"].unique(), dtype=torch.int64
        ),
        "phenotype": torch.tensor(
            nodes.loc[nodes["node_type"] == "effect/phenotype", "node_index"].unique(), dtype=torch.int64
        ),
    }
    node_type_to_z = edge.groupby("node_types")["z_index"].unique().to_dict()
    all_ids_node2 = {}
    for z in node_type_to_z.get("disease|drug", []):
        all_ids_node2[z] = all_ids["drug"]
    for z in node_type_to_z.get("drug|gene/protein", []):
        all_ids_node2[z] = all_ids["gene"]
    for z in node_type_to_z.get("drug|effect/phenotype", []):
        all_ids_node2[z] = all_ids["phenotype"]

    hp = all_param["best_hyperparam"]
    seq_len = (all_param["node2vec"]["walk_length"] * 2) - 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Hyperparams: {hp}, seq_len={seq_len}\n")

    checkpoints = sorted(glob.glob("data/model_checkpoints/saved_models_evaluation/*.pt"))
    all_results = []

    for ckpt_path in checkpoints:
        fname = os.path.basename(ckpt_path).replace(".pt", "")
        parsed = parse_checkpoint(fname)
        if parsed is None:
            print(f"Skipping (unrecognised): {fname}")
            continue

        inference, split_dir, queried_edge_types, queried_node_types, edge_flags, seed, use_dropna = parsed

        print("=" * 68)
        print(f"Checkpoint : {fname}")
        print(f"Split      : {split_dir}  |  inference={inference}  seed={seed}")

        read_csv = lambda p: pd.read_csv(p).dropna() if use_dropna else pd.read_csv(p)
        train_df = read_csv(f"data/TxGNN_splits/{split_dir}/train.csv")
        val_df   = read_csv(f"data/TxGNN_splits/{split_dir}/valid.csv")
        test_df  = read_csv(f"data/TxGNN_splits/{split_dir}/test.csv")

        type2order = dict(zip(queried_edge_types, queried_node_types))
        _, _, test_df_p = prepare_txgnn_data(nodes, train_df, val_df, test_df, type2order)

        print("Building test edges...")
        test_label_idx, test_label, edge_flag = build_test_edges(
            edge, test_df_p, queried_edge_types, queried_node_types, relation_node_types, seed
        )

        print("Loading checkpoint...")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            print(f"  SKIP — cannot load checkpoint: {e}\n")
            continue

        # Two saved formats: raw state dict (seed1) vs. wrapper dict (seeds 2-5)
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
            ckpt_hp   = ckpt.get("hyperparams", hp)
            ckpt_seq  = ckpt.get("seq_len", seq_len)
        else:
            state_dict = ckpt
            ckpt_hp   = hp
            ckpt_seq  = seq_len

        model = TransformerModel(
            vocab_size,
            ckpt_hp["hidden_channels"],
            ckpt_hp["nhead"],
            ckpt_hp["N_encoder_layers"],
            ckpt_seq,
            device=device,
            feat=feat,
            nodes=nodes,
            entity=entity,
            relation_mask_index=relation_mask_index,
            pos_emb="fixed",
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        print("Running sim()...")
        pred = sim(
            test_label_idx, model,
            batch_size=ckpt_hp["batch_size"], method="prob",
            device=device, mask_token=mask_token, seq_len=ckpt_seq,
        )
        pred = torch.tensor(pred, dtype=torch.float32)

        print("Computing AUC / AUPRC / F1...")
        lp = link_pred_val(pred, test_label, edge_flag, all_param, edge_type=edge_flags)

        print("Computing Recall@100 / Precision@100...")
        recall_k, prec_k = node_level_eval(
            test_label_idx, test_label, all_ids_node2,
            model, device=device, mask_token=mask_token, seq_len=ckpt_seq,
        )

        print(f"\n  {'Edge type':<22} {'AUC':>7} {'AUPRC':>7} {'R@100':>7} {'P@100':>7} {'F1':>7}")
        print(f"  {'-'*62}")
        for i, et in enumerate(queried_edge_types):
            auc_v   = float(lp.iloc[i]["auc"])   if i < len(lp) else float("nan")
            auprc_v = float(lp.iloc[i]["auprc"]) if i < len(lp) else float("nan")
            r = recall_k[i] if i < len(recall_k) else float("nan")
            p = prec_k[i]   if i < len(prec_k)   else float("nan")
            mask = (edge_flag == i + 1)
            y_true = test_label[mask].numpy()
            y_scores = pred[mask].numpy()
            precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)
            denom = precision_arr + recall_arr
            f1_arr = np.where(denom > 0, 2 * precision_arr * recall_arr / denom, 0.0)
            f1 = float(f1_arr.max())
            print(f"  {et:<22} {auc_v:>7.4f} {auprc_v:>7.4f} {r:>7.4f} {p:>7.4f} {f1:>7.4f}")
            all_results.append({
                "checkpoint": fname, "inference": inference, "seed": seed, "edge_type": et,
                "auc": round(auc_v, 4), "auprc": round(auprc_v, 4),
                "recall@100": round(r, 4), "precision@100": round(p, 4),
                "f1": round(f1, 4),
            })
        print()

        del model, state_dict
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    os.makedirs("data/result", exist_ok=True)
    out = "data/result/all_checkpoints_evaluation.csv"
    df.to_csv(out, index=False)
    print(f"Results saved to: {out}")
    print("\nFull results table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
