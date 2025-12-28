import os
import json
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from netmedgpt.model import TransformerModel
from netmedgpt.metapath_generation import metapath_generation
import argparse
import time

####### argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default= 0, help= "GPU device ID")
parser.add_argument("--head_index", required=False, default= [14016], help= "provide node_index of the head node")
parser.add_argument("--head_type", required=False, default= "drug", help= "provide head type")
parser.add_argument("--tail_type", required=False, default= "gene/protein", help= "provide tail type")
parser.add_argument("--relation_type", required=False, default= "drug_protein", help= "provide the relation type")
parser.add_argument("--N_top", required=False, default= 5, help= "provide the number of the top prediction for tail node")

args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
head_idx = args.head_index
head_type = args.head_type
tail_type = args.tail_type
relation_type = args.relation_type
N_top = args.N_top
batch_size = 1

#########################
with open("data/parameters.json", 'r') as file:
    all_param = json.load(file)
model_dir = os.path.join(all_param['files']['data_dir'], 'model_checkpoints')
data_dir = all_param['files']['data_dir']
user_response = os.path.join(all_param['files']['data_dir'], 'user_response')
checkpoint_path_netmedgpt = os.path.join(model_dir, "netmedgpt.pt")

#########################
# load data
nodes = pd.read_csv(os.path.join(data_dir, 'KG/nodes_preprocessed.csv'), sep= ',')
edge = pd.read_csv(os.path.join(data_dir, "KG/edges_preprocessed.csv")) 
feat = torch.load(os.path.join(data_dir, "KG/embeddings_preprocessedKG.pt"))

# identify the edge indecies and concat with node indices
index_relations = edge[['relation', 'z_index', 'relation']].drop_duplicates()
index_relations.columns = ['node_type', 'node_index', 'node_name']
node_edge_indecices = pd.concat([nodes, index_relations])
mask_token_id = edge['z_index'].max() +1
relation_token_id = edge.loc[edge['relation'] == relation_type,'z_index'].drop_duplicates().values.item()

#############
# for model instatiation
all_relation_type = list(edge['relation'].unique())
node_types = list(nodes['node_type'].unique())
mask = ['mask']
entity = node_types + all_relation_type + mask
relation_index = edge.loc[edge['relation'].isin(all_relation_type), ['relation', 'z_index']].drop_duplicates()
mask_row = pd.DataFrame([['mask', mask_token_id]], columns=['relation', 'z_index'])
relation_mask_index = pd.concat([relation_index, mask_row], ignore_index=True)
vocab_size = mask_token_id + 1

#########################
# load model
checkpoint = torch.load(checkpoint_path_netmedgpt, map_location=device)
param = checkpoint['parameters']
state_dict = checkpoint['model_state_dict']
model = TransformerModel(
    vocab_size,
    param['hidden_channels'],
    param['nhead'],
    param['N_encoder_layers'],
    (param['walk_length']*2)-1,
    device = device,
    feat = feat,
    nodes = nodes,
    entity = entity,
    relation_mask_index = relation_mask_index,
    pos_emb='fixed',
).to(device)
model.load_state_dict(state_dict)

##############################################################################################################
thr = 0.8
n_metapath = 10

nodes_at_mask = nodes.loc[nodes['node_type'] == tail_type,['node_index','node_name']].values
node_ids_at_mask = torch.tensor(nodes_at_mask[:,0].astype(int)).to(device)
index_name_map = dict(zip(node_edge_indecices["node_index"], node_edge_indecices["node_name"]))
index_type_map = dict(zip(node_edge_indecices["node_index"], node_edge_indecices["node_type"]))
relation_edges = set(edge.loc[(edge['relation'] == relation_type) & (edge['node_types'] == f"{head_type}|{tail_type}"), 'xy'])

########
### generate walk with head and relation relation to give to the model for finding top_k tails
sentence = torch.ones(len(head_idx), param['walk_length'] * 2 - 1) * mask_token_id
sentence[:, 0] = torch.tensor(head_idx, dtype=torch.long)
sentence[:, 1] = relation_token_id
sentence = sentence.to(torch.long).to(device)

dataset = TensorDataset(sentence)
dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=False)
mask_pos = torch.where(sentence == mask_token_id)[1][0].item()
top_nodes_at_mask_all = []
top_probs_nodes_at_mask_all = []

for batch in tqdm(dataloader):
    input_batch = batch[0].to(device)
    with torch.no_grad():
        output = model(input_batch)
        logits = output[:, mask_pos, node_ids_at_mask]
        probs = F.softmax(logits, dim=1)

        N = min(N_top, probs.size(1))
        top_probs, top_idx = torch.topk(probs, N, dim=1)

        tail_ids = node_ids_at_mask[top_idx]
        top_nodes_at_mask_all.append(tail_ids) 
        top_probs_nodes_at_mask_all.append(top_probs)  

top_nodes_at_mask_all = torch.cat(top_nodes_at_mask_all, dim=0)
top_probs_nodes_at_mask_all = torch.cat(top_probs_nodes_at_mask_all, dim=0)

##############################
start = time.time()
i = 0
all_triplets = []
for j in tqdm(range(top_nodes_at_mask_all.shape[1])):
    head_index = head_idx[i]
    tail_index = top_nodes_at_mask_all[i, j].item()
    is_existing_head_tail = f'{head_index}|{tail_index}' in relation_edges

    walk_input1 = [head_index, relation_token_id, tail_index] + [mask_token_id] * 6
    walk_input2 = [tail_index, relation_token_id, head_index] + [mask_token_id] * 6

    walk_input1 = torch.tensor([walk_input1], device=device)
    walk_input2 = torch.tensor([walk_input2], device=device)

    metapath1, probs1_scaled, probs1 = metapath_generation(walk_input1, n_metapath, mask_token_id, model, node_edge_indecices)
    metapath2, probs2_scaled, probs2 = metapath_generation(walk_input2, n_metapath, mask_token_id, model, node_edge_indecices)

    metapath = torch.cat(metapath1 + metapath2, dim=0).detach().cpu().numpy()
    probs = torch.cat(probs1 + probs2, dim=0).detach().cpu().numpy()
    probs_scaled = torch.cat(probs1_scaled + probs2_scaled, dim=0).detach().cpu().numpy()

    triplets = []
    prob_triplets = []
    prob_scaled_triplets = []

    for row, prob_scaled, prob in zip(metapath, probs_scaled, probs):
        for k in range(0, len(row) - 2, 2):
            triplets.append(row[k:k+3].tolist())
            prob_triplets.append(prob[k:k+3].tolist())
            prob_scaled_triplets.append(prob_scaled[k:k+3].tolist())

    triplet_df = pd.DataFrame(triplets, columns=["head_index", "relation_index", "tail_index"])
    prob_triplets_df = pd.DataFrame(prob_triplets, columns=["head_prob", "relation_prob", "tail_prob"])
    prob_scaled_triplets_df = pd.DataFrame(prob_scaled_triplets, columns=["head_prob_scaled", "relation_prob_scaled", "tail_prob_scaled"])

    triplet_df = pd.concat([triplet_df, prob_triplets_df, prob_scaled_triplets_df], axis=1)
    triplet_df["head_type"] = triplet_df["head_index"].map(index_type_map)
    triplet_df["relation_type"] = triplet_df["relation_index"].map(index_type_map)
    triplet_df["tail_type"] = triplet_df["tail_index"].map(index_type_map)
    triplet_df["head_name"] = triplet_df["head_index"].map(index_name_map)
    triplet_df["tail_name"] = triplet_df["tail_index"].map(index_name_map)
    triplet_df["pair_id"] = f"{head_index}|{tail_index}"
    triplet_df["drug_disease_in_KG"] = is_existing_head_tail
    all_triplets.append(triplet_df)


final_df = pd.concat(all_triplets, ignore_index=True)
final_df.to_csv(os.path.join(user_response, f"subnetwork.csv"), index=False)

end = time.time()
single_time = (end - start)/60
print(f"Time per iteration: {single_time:.3f} mins")
    
