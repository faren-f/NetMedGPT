import os
import subprocess
import json
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
import argparse

from src.utilities import link_pred_val, node_level_eval, sim 
from src.model import TransformerModel, create_mask, lr_lambda
from src.prepare_txgnn_splits import prepare_txgnn_data
from src.sentence_generation import sentence_generation

####### arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=False, default= 0, help="GPU device ID")
parser.add_argument("--seed", required=False, default= 1, help="Random seed")
parser.add_argument("--inference", required=False, default= "random_link_split", help="please select among: random_link_split, zero_shot_split, adrenal_gland, anemia, autoimmune, cardiovascular, cell_proliferation, diabetes, mental_health, metabolic_disorder, neurodigenerative")

args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
seed = args.seed
inference = args.inference

Epoch = 1#100
patience_limit = 10
with open("data/parameters.json", 'r') as file:
    all_param = json.load(file)

data_dir = all_param['files']['data_dir']
best_hyperparam = all_param['best_hyperparam']

os.makedirs(os.path.join(all_param['files']['data_dir'], 'result'), exist_ok = True) 
result_dir = os.path.join(all_param['files']['data_dir'], 'result')

os.makedirs(os.path.join(all_param['files']['data_dir'], "saved_models"), exist_ok=True)
model_dir = os.path.join(all_param['files']['data_dir'], 'saved_models')

############################################################################################################################
nodes = pd.read_csv(os.path.join(data_dir, 'KG/nodes.csv'), sep= ',')
edge = pd.read_csv(os.path.join(data_dir, "KG/edges.csv")) 
feat = torch.load(os.path.join(data_dir, "KG/embeddings.pt"))

#######################################################################################################################
if inference == "random_link_split":
    queried_edge_types = ['indication', 'contraindication', 'off-label use', 'drug_protein', 'drug_effect']
    queried_node_types = ['disease|drug', 'disease|drug', 'disease|drug', 'drug|gene/protein', 'drug|effect/phenotype']
    edge_flags_val = [1, 2, 3, 4, 5]
    edge_flags_test = [1, 2, 3, 4, 5]
    train_txgnn = pd.read_csv(f"data/TxGNN_splits/random_{seed}/train.csv")
    val_txgnn = pd.read_csv(f"data/TxGNN_splits/random_{seed}/valid.csv")
    test_txgnn = pd.read_csv(f"data/TxGNN_splits/random_{seed}/test.csv")
    
elif inference == "zero_shot_split":
    queried_edge_types = ['indication', 'contraindication']
    queried_node_types = ['disease|drug', 'disease|drug']
    edge_flags_val = [1, 2]
    edge_flags_test = [1, 2]
    train_txgnn = pd.read_csv(f"data/TxGNN_splits/complex_disease_{seed}/train.csv")
    val_txgnn = pd.read_csv(f"data/TxGNN_splits/complex_disease_{seed}/valid.csv")
    test_txgnn = pd.read_csv(f"data/TxGNN_splits/complex_disease_{seed}/test.csv")

else:
    area = inference
    queried_edge_types = ['indication', 'contraindication']
    queried_node_types = ['disease|drug', 'disease|drug']
    edge_flags_val = [1, 2]
    edge_flags_test = [1, 2]
    train_txgnn = pd.read_csv(f"data/TxGNN_splits/{area}_{seed}/train.csv").dropna()
    val_txgnn = pd.read_csv(f"data/TxGNN_splits/{area}_{seed}/valid.csv").dropna()
    test_txgnn = pd.read_csv(f"data/TxGNN_splits/{area}_{seed}/test.csv").dropna()

type2order = dict(zip(queried_edge_types, queried_node_types))
train_txgnn, val_txgnn, test_txgnn = prepare_txgnn_data(nodes, train_txgnn, val_txgnn, test_txgnn, type2order)

with open(f"{data_dir}/KG/relation_node_types.pkl", "rb") as f:
    relation_node_types = pickle.load(f)

if inference == "disease_area_split":
    all_ids_node2 = nodes.loc[nodes['node_type'].isin(['drug']),'node_index'].unique()
    all_ids_node2 = torch.tensor(sorted(all_ids_node2), dtype=torch.int64)

else:
    all_ids = {
        'drug': torch.tensor(
            nodes.loc[nodes['node_type'] == 'drug', 'node_index'].unique(), dtype=torch.int64
        ),
        'gene': torch.tensor(
            nodes.loc[nodes['node_type'] == 'gene/protein', 'node_index'].unique(), dtype=torch.int64
        ),
        'phenotype': torch.tensor(
            nodes.loc[nodes['node_type'] == 'effect/phenotype', 'node_index'].unique(), dtype=torch.int64
        )
    }
    
    
    node_type_to_z = edge.groupby('node_types')['z_index'].unique().to_dict()
    
    all_ids_node2 = {}
    for z in node_type_to_z.get('disease|drug', []):
        all_ids_node2[z] = all_ids['drug']
    
    for z in node_type_to_z.get('drug|gene/protein', []):
        all_ids_node2[z] = all_ids['gene']
    
    for z in node_type_to_z.get('drug|effect/phenotype', []):
        all_ids_node2[z] = all_ids['phenotype']
    
    
mask_token = edge['z_index'].max() +1
vocab_size = mask_token + 1

relation_type = list(edge['relation'].unique())
node_types = list(nodes['node_type'].unique())

mask = ['mask']
entity = node_types + relation_type + mask

relation_index = edge.loc[edge['relation'].isin(relation_type), ['relation', 'z_index']].drop_duplicates()
mask_row = pd.DataFrame([['mask', mask_token]], columns=['relation', 'z_index'])
relation_mask_index = pd.concat([relation_index, mask_row], ignore_index=True)

patience_counter = 0
data = sentence_generation(edge, nodes, queried_node_types, queried_edge_types, train_txgnn, val_txgnn, test_txgnn, relation_node_types, all_param, seed)

node2vec_walks = data['node2vec_walks']
val_data_edge_label = data['val_data_edge_label']
val_data_edge_label_index = data['val_data_edge_label_index_with_relation']
edge_type_flag_val = data['edge_type_flag_val']

test_data_edge_label = data['test_data_edge_label']
test_data_edge_label_index = data['test_data_edge_label_index_with_relation']
edge_type_flag_test = data['edge_type_flag_test']

log_name = f"netmedgpt_{inference}_{seed}" 

model_save_path = f"{model_dir}/{log_name}.pt"
seq_len = (all_param['node2vec']['walk_length']*2)-1   

dataset = TensorDataset(node2vec_walks)
dataloader = DataLoader(dataset, batch_size=best_hyperparam['batch_size'], shuffle=True)

model = TransformerModel(
    vocab_size,
    best_hyperparam['hidden_channels'],
    best_hyperparam['nhead'],
    best_hyperparam['N_encoder_layers'],
    (all_param['node2vec']['walk_length']*2)-1, 
    device=device,
    feat = feat,
    nodes = nodes,
    entity = entity,
    relation_mask_index = relation_mask_index,
    pos_emb='fixed',
).to(device)

optimizer = optim.Adam(model.parameters(), lr=best_hyperparam['learning_rate'])
scaler = GradScaler()
warmup_steps = int(0.1 * Epoch * len(dataloader))  
total_steps = Epoch * len(dataloader)
# scheduler = LambdaLR(optimizer, lr_lambda)  
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps, total_steps))

criterion = nn.CrossEntropyLoss(ignore_index=vocab_size)

best_auprc = 0.0
best_model_state = None
for epoch in range(Epoch):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        input_batch = batch[0].to(device)
        masked_input, mask = create_mask(input_batch, vocab_size, mask_token_id=mask_token)
    
        with autocast():
            output = model(masked_input)
            loss = criterion(output[mask], input_batch[mask])
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        batch_loss = loss.item()
        epoch_loss += batch_loss

    avg_loss = epoch_loss / len(dataloader)
    pred_val = sim(val_data_edge_label_index, model, batch_size=best_hyperparam['batch_size'], method='prob', device=device, mask_token=mask_token, seq_len=seq_len)
    pred_val = torch.tensor(pred_val, dtype=torch.float32)
    LP_result_val = link_pred_val(pred_val, val_data_edge_label, edge_type_flag_val, all_param, edge_type = edge_flags_val)
    hit_k_val, precision_k_val = node_level_eval(val_data_edge_label_index, val_data_edge_label, all_ids_node2, model, device=device, mask_token=mask_token, seq_len=seq_len)

    auprc = np.array(LP_result_val['auprc']).mean()
    print(auprc)
    
    if (auprc > best_auprc):
        best_auprc = auprc
        best_model_state = model.state_dict()
        best_optimizer_state = optimizer.state_dict()
        best_epoch = epoch + 1
        best_param = best_hyperparam
        loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"patience_counter is {patience_counter}")


    pred_test = sim(test_data_edge_label_index, model, batch_size=best_hyperparam['batch_size'], method='prob', device=device, mask_token=mask_token, seq_len=seq_len)
    pred_test = torch.tensor(pred_test, dtype=torch.float32)
    
    LP_result_test = link_pred_val(pred_test, test_data_edge_label, edge_type_flag_test, all_param, edge_type=edge_flags_test)
    hit_k_test, precision_k_test = node_level_eval(test_data_edge_label_index, test_data_edge_label, all_ids_node2, model, device=device, mask_token=mask_token, seq_len=seq_len)

    if (patience_counter >= patience_limit):
            break    

#save best model
# if best_model_state:
#     torch.save(best_model_state, model_save_path)

if best_model_state:
    ckpt = {
        "model_state": best_model_state,
        "optimizer_state": best_optimizer_state,
        "scaler_state": scaler.state_dict(),
        "best_epoch": best_epoch,
        "best_auprc": best_auprc,
        "hyperparams": best_hyperparam,
        "all_param": all_param,
        "seed": seed,
        "seq_len": seq_len,
        'loss': loss,
        "vocab_size": vocab_size,
    }
    torch.save(ckpt, model_save_path)
    

# Save final val result
columns = ['auc', 'auprc']
LP_result_val_df = pd.DataFrame(LP_result_val, columns=columns)
LP_result_val_df.index = queried_edge_types
LP_result_val_df['seed'] = seed

hit_val = pd.DataFrame(hit_k_val, columns=['hit'])
precision_val = pd.DataFrame(precision_k_val, columns=['precision'])
hit_val.index = queried_edge_types
precision_val.index = queried_edge_types
hit_precision_val = pd.concat([hit_val, precision_val], axis=1)
hit_precision_val['seed'] = seed

# save final test result 
LP_result_test_df = pd.DataFrame(LP_result_test, columns=columns)
LP_result_test_df.index = queried_edge_types
LP_result_test_df['seed'] = seed

hit_test = pd.DataFrame(hit_k_test, columns=['hit'])
precision_test = pd.DataFrame(precision_k_test, columns=['precision'])
hit_test.index = queried_edge_types
precision_test.index = queried_edge_types

hit_precision_test = pd.concat([hit_test, precision_test], axis=1)
hit_precision_test['seed'] = seed

config_dir = os.path.join(result_dir, f"netmedgpt_{inference}")

os.makedirs(config_dir, exist_ok=True)

LP_result_val_df.to_csv(os.path.join(config_dir, f"LP_val.csv"), index=True)
hit_precision_val.to_csv(os.path.join(config_dir, f"hit_precision_val.csv"), index=True)
LP_result_test_df.to_csv(os.path.join(config_dir, f"LP_test.csv"), index=True)
hit_precision_test.to_csv(os.path.join(config_dir, f"hit_precision_test.csv"), index=True)  

