"""
Drug and ADR features for the DeepADR baseline (deepadr_baseline.py), as described in the DeepADR paper
(Wang et al., bioRxiv 10.1101/2025.07.29.667409):
  - drug structure : MoLFormer embedding of the SMILES (ibm/MoLFormer-XL-both-10pct, version 2024-03-31, pooled output, 768-d)
  - ADR semantics  : BioBERT [CLS] embedding of the ADR term (dmis-lab/biobert-base-cased-v1.1, 768-d)
The drug-target multi-hot vector is built in deepadr_baseline.py from the training drug_protein edges.

The DeepADR repository only contains these features for its own 719 SIDER drugs and 994 ADR terms, so they are
computed here in the same way for the KG drugs (SMILES from dti_inputs.py) and all effect/phenotype nodes
(node_name = HPO term). Independent of split and seed; saves:

  data/new_analysis/adr_inputs/drug_molformer.pt     {'node_index': LongTensor, 'emb': FloatTensor (n, 768)}
  data/new_analysis/adr_inputs/phenotype_biobert.pt  {'node_index': LongTensor, 'emb': FloatTensor (n, 768)}

Drugs without SMILES (mostly biologics) are not in drug_molformer.pt.

Example (from the repository root, in the deepadr environment, after dti_inputs.py):
    python adr_inputs.py --gpu 0
"""

import os
import argparse
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default=0, type=int, help="GPU device ID")
parser.add_argument("--out_dir", required=False, default="data/new_analysis/adr_inputs")
parser.add_argument("--smiles_csv", required=False, default="data/new_analysis/dti_inputs/drug_smiles.csv")
parser.add_argument("--nodes", required=False, default="data/KG/nodes.csv")
parser.add_argument("--batch_size", required=False, default=64, type=int)

MOLFORMER = "ibm/MoLFormer-XL-both-10pct"
# model version of 2024-03-31 (same weights); later versions of its code need transformers >= 4.56
MOLFORMER_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"
BIOBERT = "dmis-lab/biobert-base-cased-v1.1"


@torch.no_grad()
def embed(texts, model, tokenizer, device, batch_size, pool, max_length):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_length,
                          return_tensors="pt").to(device)
        res = model(**batch)
        out.append((res.pooler_output if pool == "pooler" else res.last_hidden_state[:, 0]).float().cpu())
    return torch.cat(out)


def main():
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # drugs: MoLFormer on the SMILES from dti_inputs.py
    drugs = pd.read_csv(args.smiles_csv).dropna(subset=['smiles'])
    tok = AutoTokenizer.from_pretrained(MOLFORMER, revision=MOLFORMER_REVISION, trust_remote_code=True)
    model = AutoModel.from_pretrained(MOLFORMER, revision=MOLFORMER_REVISION, deterministic_eval=True,
                                      trust_remote_code=True).to(device).eval()
    emb = embed(drugs['smiles'].tolist(), model, tok, device, args.batch_size, pool="pooler", max_length=512)
    torch.save({'node_index': torch.tensor(drugs['node_index'].values), 'emb': emb},
               os.path.join(args.out_dir, 'drug_molformer.pt'))
    print(f"MoLFormer: {emb.shape[0]} drugs, dim {emb.shape[1]}")
    del model

    # ADRs: BioBERT [CLS] on the phenotype names
    nodes = pd.read_csv(args.nodes)
    phen = nodes[nodes['node_type'] == 'effect/phenotype']
    tok = AutoTokenizer.from_pretrained(BIOBERT)
    model = AutoModel.from_pretrained(BIOBERT).to(device).eval()
    emb = embed(phen['node_name'].astype(str).tolist(), model, tok, device, args.batch_size, pool="cls", max_length=64)
    torch.save({'node_index': torch.tensor(phen['node_index'].values), 'emb': emb},
               os.path.join(args.out_dir, 'phenotype_biobert.pt'))
    print(f"BioBERT: {emb.shape[0]} phenotypes, dim {emb.shape[1]}")


if __name__ == "__main__":
    main()
