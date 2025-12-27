import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import warnings

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False*")

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, N_encoder_layers, seq_len, device, feat, nodes, entity, relation_mask_index, mask_token_id=None, pos_emb=None):
        super(TransformerModel, self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.nodes = nodes
        self.entity = entity
        self.relation_mask_index = relation_mask_index

        ## MLP configuration for node attributes    
        self.feat_all = nn.ParameterDict()
        self.fc_all = nn.ModuleDict()
        for k, v in feat.items():
            if len(v) > 1:                      
                for k1, v1 in v.items():
                    name = f"{k}|{k1}"  
                    self.feat_all[name] = nn.Parameter(v1.to(device), requires_grad = False)
                    self.fc_all[name] = nn.Sequential(
                        nn.Linear(v1.shape[1], d_model, device=device)
                    )
                    
            else:
                v1 = list(v.values())[0]
                param = nn.Parameter(v1.to(device))
                if list(v.keys())[0] == 'random':  
                    param.requires_grad = True
                elif list(v.keys())[0] == 'fixed':    
                    param.requires_grad = False
                else:                                
                    param.requires_grad = False
                    
                    
                self.feat_all[k] = param
                self.fc_all[k] = nn.Sequential(
                        nn.Linear(v1.shape[1], d_model, device=device)
                    )


        self.pos_emb = pos_emb # positional embedding
        if pos_emb == 'fixed':
            self.pos_encoding = self.create_positional_encoding(seq_len, d_model).to(device)    # Positional encoding (computed dynamically)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, activation='gelu').to(device),
            num_layers = N_encoder_layers
            )

        
        self.fc = nn.Linear(d_model, vocab_size+1).to(device)  
        
        # binary classification layer 
        self.classifire = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  
            nn.ReLU(),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model//3),
            nn.ReLU(),
            nn.Linear(d_model//3, 1)
        ).to(device)

        
    def forward(self, x):  
        x = self.compute_embedding(x) + self.get_x_pos_emb(x)
        x = self.transformer(x.permute(1, 0, 2))    
        x = self.fc(x.permute(1, 0, 2))
        return x

    def classify(self, x):   
        x = self.compute_embedding(x) + self.get_x_pos_emb(x)
        x = self.transformer(x.permute(1, 0, 2))
        x = x.permute(1, 0, 2)
        x = x[:, 0:3:2, :]            
        x = x.reshape(x.size(0), -1) 
        x = self.classifire(x)
        return x

    def get_embeddings(self, tokens, select = 'raw'):
        with torch.no_grad():
            if select == 'raw':
                embeddings = self.compute_embedding(tokens)
            elif select == 'transformer':
                x = self.compute_embedding(tokens) + self.get_x_pos_emb(tokens)
                x = self.transformer(x.permute(1, 0, 2))
                embeddings = x.permute(1, 0, 2)
            else:
                raise(f'invalid "select" option: {select}')
        return embeddings

    # when we use node attr we generate embedding here with trainable parameters 
    def compute_embedding(self, tokens):
        W_list = []
        Indices = []
        for t in self.entity:
            matching_keys = [k for k in self.feat_all.keys() if k.split('|')[0]==t]
            tensors = [self.fc_all[k](self.feat_all[k]) for k in matching_keys]
            merged_tensor = torch.stack(tensors, dim=0).sum(dim=0)  # or use torch.mean(...), etc.
            W_list.append(merged_tensor)
            indices = list(self.nodes.loc[self.nodes['node_type'] == t,'node_index'])
            if len(indices)>0:
                Indices = Indices + indices
            elif len(indices) == 0:
                index = list(self.relation_mask_index.loc[self.relation_mask_index['relation'] == t, 'z_index'])
                Indices = Indices + index

        W = torch.cat(W_list, dim=0).to(self.device)
        entity_indices = torch.tensor(Indices)
        
        # Get the sorted indices (i.e., the order to sort by)
        sorted_order = torch.argsort(entity_indices)
        
        # Apply the sorted order to both the indices and the tensor W
        sorted_W = W[sorted_order]

        return F.embedding(tokens, sorted_W)

    
    def create_positional_encoding(self, seq_len, d_model):
        """Generates positional encoding dynamically"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def get_x_pos_emb(self, x):
        if self.pos_emb == 'fixed':
            x_pos_emb = self.pos_encoding[:x.shape[1], :]
        else:
            x_pos_emb = 0
        return x_pos_emb


def create_mask(data, vocab_size, mask_token_id, mask_prob=0.2, min_mask = 4):
    masked_data = data.clone()
    mask = torch.rand(data.shape) < mask_prob  
    
    for i in range(masked_data.shape[0]):
        if mask[i].sum() < min_mask:
            mask[i] = torch.zeros(data.shape[1], dtype=torch.bool)
            true_indices = torch.randperm(data.shape[1])[:min_mask]    
            mask[i, true_indices] = True
        elif mask[i].sum() == data.shape[1]:                           
            mask[i, 0] = False
    masked_data[mask] = mask_token_id  
    return masked_data, mask

########################################################################################################
class Classifier(nn.Module):
    def __init__(self, d_model, d_classifier, device):
        super(Classifier, self).__init__()
        
        # binary classification layer 
        layers = []
        input_dim = d_model * 2  
        for output_dim in d_classifier:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, 1))  
        self.classifier = nn.Sequential(*layers).to(device)

    def forward(self, x):
        # x = x[:, :3, :]              
        x = x[:, 0:3:2, :]              
        x = x.reshape(x.size(0), -1) 
        x = self.classifier(x)
        return x

def lr_lambda(current_step, warmup_steps, total_steps):
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))







