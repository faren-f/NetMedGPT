"""
Drug SMILES and protein sequences for the sequence-based DTI baselines (PSICHIC, ...).

The KG drug nodes are DrugBank IDs and the gene/protein nodes are NCBI Gene IDs. Sequence-based DTI methods
need a SMILES string per drug and an amino-acid sequence per protein, so this script maps them once
(independent of split and seed) and saves:

  data/new_analysis/dti_inputs/drug_smiles.csv        node_index, node_id (DrugBank), smiles, cid
  data/new_analysis/dti_inputs/protein_sequences.csv  node_index, node_id (NCBI Gene), node_name, uniprot, sequence

Sources:
  SMILES   : PubChem PUG REST, DrugBank ID as registry ID (property "SMILES", isomeric); the first CID is used.
             Biologics (antibodies, peptides, enzymes, ...) usually have no PubChem compound and stay without SMILES.
             SMILES that RDKit cannot parse are dropped (if RDKit is installed).
  Sequence : UniProt REST, all reviewed (Swiss-Prot) human entries; NCBI Gene ID -> canonical sequence.
             Gene IDs not found are matched by the primary gene symbol (KG node_name).
             If several Swiss-Prot entries share a Gene ID, the entry with a single Gene ID, then the longest
             sequence is used.

Nodes without SMILES / sequence cannot be scored by sequence-based methods (see psichic_baseline.py).
Already downloaded SMILES are kept, so the script can be restarted after a stop.

Example (from the repository root):
    python dti_inputs.py
"""

import os
import io
import time
import argparse
import requests
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=False, default="data/new_analysis/dti_inputs")
parser.add_argument("--nodes", required=False, default="data/KG/nodes.csv")
parser.add_argument("--sleep", required=False, default=0.25, type=float, help="seconds between PubChem requests (limit 5/s)")

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/{}/property/SMILES/CSV"
UNIPROT = ("https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true"
           "&fields=accession,xref_geneid,gene_primary,sequence&format=tsv")


#######################################################################################################################
# Drugs: DrugBank ID -> SMILES

# one PubChem request per DrugBank ID; returns (smiles, cid) or (None, None) if PubChem has no compound
def pubchem_smiles(drugbank_id, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(PUBCHEM.format(drugbank_id), timeout=30)
            if r.status_code == 404:
                return None, None
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                if len(df) and isinstance(df.loc[0, 'SMILES'], str):
                    return df.loc[0, 'SMILES'], int(df.loc[0, 'CID'])
                return None, None
        except (requests.RequestException, pd.errors.ParserError):
            pass
        time.sleep(2 * (attempt + 1))  # server busy (503) or connection problem
    return None, None


def valid_smiles(smiles):
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except ImportError:
        return True
    return Chem.MolFromSmiles(smiles) is not None


def drug_smiles(nodes, out_csv, sleep):
    drugs = nodes.loc[nodes['node_type'] == 'drug', ['node_index', 'node_id']].reset_index(drop=True)

    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=['node_index', 'node_id', 'smiles', 'cid'])
    todo = drugs[~drugs['node_id'].isin(done['node_id'])]
    print(f"drugs: {len(drugs)} | already done: {len(done)} | to query: {len(todo)}")

    rows = []
    for i, (idx, db) in enumerate(zip(todo['node_index'], todo['node_id'])):
        smiles, cid = pubchem_smiles(db)
        if smiles is not None and not valid_smiles(smiles):
            smiles, cid = None, None
        rows.append({'node_index': idx, 'node_id': db, 'smiles': smiles, 'cid': cid})
        time.sleep(sleep)
        if (i + 1) % 500 == 0 or i + 1 == len(todo):  # save regularly, so a restart continues from here
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(out_csv, index=False)
            rows = []
            print(f"  {i + 1}/{len(todo)} queried")

    print(f"drugs with SMILES: {done['smiles'].notna().sum()} / {len(drugs)}")
    return done


#######################################################################################################################
# Proteins: NCBI Gene ID -> UniProt sequence

def protein_sequences(nodes, out_csv):
    r = requests.get(UNIPROT, timeout=600)
    r.raise_for_status()
    up = pd.read_csv(io.StringIO(r.text), sep='\t').rename(columns={
        'Entry': 'uniprot', 'GeneID': 'gene_ids', 'Gene Names (primary)': 'symbol', 'Sequence': 'sequence'})
    print(f"UniProt reviewed human entries: {len(up)}")

    # one row per (Gene ID, entry); entries with a single Gene ID first, then the longest sequence
    up['n_gene_ids'] = up['gene_ids'].fillna('').str.count(';')
    up['length'] = up['sequence'].str.len()
    up = up.sort_values(['n_gene_ids', 'length'], ascending=[True, False])
    by_id = up.assign(gene_id=up['gene_ids'].fillna('').str.split(';')).explode('gene_id')
    by_id = by_id[by_id['gene_id'] != ''].drop_duplicates('gene_id').set_index('gene_id')
    by_symbol = up.dropna(subset=['symbol']).drop_duplicates('symbol').set_index('symbol')

    genes = nodes.loc[nodes['node_type'] == 'gene/protein', ['node_index', 'node_id', 'node_name']].reset_index(drop=True)
    uniprot, sequence = [], []
    for gid, name in zip(genes['node_id'].astype(str), genes['node_name']):
        if gid in by_id.index:
            hit = by_id.loc[gid]
        elif name in by_symbol.index:
            hit = by_symbol.loc[name]
        else:
            hit = None
        uniprot.append(None if hit is None else hit['uniprot'])
        sequence.append(None if hit is None else hit['sequence'])
    genes['uniprot'] = uniprot
    genes['sequence'] = sequence
    genes.to_csv(out_csv, index=False)

    print(f"proteins with sequence: {genes['sequence'].notna().sum()} / {len(genes)}")
    return genes


def main():
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    nodes = pd.read_csv(args.nodes)

    protein_sequences(nodes, os.path.join(args.out_dir, 'protein_sequences.csv'))
    drug_smiles(nodes, os.path.join(args.out_dir, 'drug_smiles.csv'), args.sleep)


if __name__ == "__main__":
    main()
