import os
import json
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from netmedgpt.model import TransformerModel, Classifier
import argparse
import time

####### argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default= 0, type=int, help= "GPU device ID")
parser.add_argument("--head_csv", required=True, help="CSV file containing disease node indices")
parser.add_argument("--batch_size", required=False, default= 1, type=int, help= "provide batch size: it should be less than number of the provided disaeses")
parser.add_argument("--N_top", required=False, default= 20, type=int, help= "provide the number of the top prediction for drug nodes")

args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
heads_df = pd.read_csv(args.head_csv)
head_idx = heads_df["head_index"].astype(int).tolist()
batch_size = args.batch_size
N_top = args.N_top

#########################
with open("data/parameters.json", 'r') as file:
    all_param = json.load(file)
model_dir = os.path.join(all_param['files']['data_dir'], 'model_checkpoints')
data_dir = all_param['files']['data_dir']
user_response = os.path.join(all_param['files']['data_dir'], 'user_response')
os.makedirs(user_response, exist_ok=True)
checkpoint_path_netmedgpt = os.path.join(model_dir, "netmedgpt.pt")
checkpoint_path_LP_indication = os.path.join(model_dir, "netmedgpt_LP_indication.pt")

#########################
# load data
nodes = pd.read_csv(os.path.join(data_dir, 'KG/nodes_preprocessed.csv'), sep= ',')
edge = pd.read_csv(os.path.join(data_dir, "KG/edges_preprocessed.csv")) 
feat = torch.load(os.path.join(data_dir, "KG/embeddings_preprocessedKG.pt"))

# identify the edge indecies and concate with node indices
head_type = "disease"
tail_type = "drug"
relation_type = "indication"

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
relation_index_all = edge.loc[edge['relation'].isin(all_relation_type), ['relation', 'z_index']].drop_duplicates()
mask_row = pd.DataFrame([['mask', mask_token_id]], columns=['relation', 'z_index'])
relation_mask_index = pd.concat([relation_index_all, mask_row], ignore_index=True)
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
# load the model: LP head–indication
checkpoint_LP_indication = torch.load(checkpoint_path_LP_indication, map_location=device)
param_LP_indication = checkpoint_LP_indication['parameters']
model_classifier_indication = Classifier(param_LP_indication['hidden_channels_encoder'], param_LP_indication['d_classifier'], device)
state_dict_LP_indication = checkpoint_LP_indication['classifier_state_dict']
model_classifier_indication.load_state_dict(state_dict_LP_indication)

######################
thr = 0.8
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

print(f"shape is {top_nodes_at_mask_all.shape}")

##############################
all_triplets = []  
for i in tqdm(range(top_nodes_at_mask_all.shape[0]), desc="Diseases"):
    start = time.time()

    for j in range(top_nodes_at_mask_all.shape[1]):
        head_index = head_idx[i]
        tail_index = top_nodes_at_mask_all[i, j].item()
        is_existing_head_tail = f'{head_index}|{tail_index}' in relation_edges
        
        sen = torch.ones(param['walk_length'] * 2 - 1) * mask_token_id
        sen[0] = torch.tensor(head_index, dtype=torch.long)
        sen[1] = relation_token_id
        sen[2] = torch.tensor(tail_index, dtype=torch.long)
        sen = sen.to(torch.long).to(device)
        
        sen = sen.unsqueeze(0)  
        emb = model.get_embeddings(sen, select = 'transformer')
        y_hat_LP_indication = model_classifier_indication(emb)
        prob_LP_indication = F.sigmoid(y_hat_LP_indication).item()
                
        if (prob_LP_indication < thr):
            continue
        
        triplet_df = pd.DataFrame([[head_index, relation_token_id, tail_index]], columns=["head_index", "relation_index", "tail_index"])
        triplet_df["head_type"] = triplet_df["head_index"].map(index_type_map)
        triplet_df["relation_type"] = triplet_df["relation_index"].map(index_type_map)
        triplet_df["tail_type"] = triplet_df["tail_index"].map(index_type_map)
        triplet_df["head_name"] = triplet_df["head_index"].map(index_name_map)
        triplet_df["tail_name"] = triplet_df["tail_index"].map(index_name_map)
        triplet_df["pair_id"] = f"{head_index}|{tail_index}"
        triplet_df["drug_disease_score_indication"] = prob_LP_indication
        triplet_df["drug_disease_in_KG"] = is_existing_head_tail
        all_triplets.append(triplet_df)

final_df = pd.concat(all_triplets, ignore_index=True)
final_df.to_csv(os.path.join(user_response, "user_response_multi_head.csv"), index=False)

end = time.time()
single_time = (end - start)/60
print(f"Time per iteration: {single_time:.3f} mins")


