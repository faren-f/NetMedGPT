import torch
import torch.nn as nn
import torch.nn.functional as F 

# Create a transformer model for masked word prediction
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
            if len(v) > 1:                      # nodes that have two or more type of attributes e.g., protein  
                for k1, v1 in v.items():
                    name = f"{k}|{k1}"  # attach parent name to subkey
                    self.feat_all[name] = nn.Parameter(v1.to(device), requires_grad = False)
                    self.fc_all[name] = nn.Sequential(
                        nn.Linear(v1.shape[1], d_model, device=device)
                        # nn.ReLU(),
                        # nn.Dropout(0.2),
                        # nn.Linear(3*d_model, 2*d_model, device=device),
                        # nn.ReLU(),
                        # nn.Linear(2*d_model, d_model, device=device)
                    )
                    
            else:
                v1 = list(v.values())[0]
                param = nn.Parameter(v1.to(device))
                if list(v.keys())[0] == 'random':  # nodes that do not have attr e.g., anatomy
                    param.requires_grad = True
                elif list(v.keys())[0] == 'fixed':    # relations and mask
                    param.requires_grad = False
                else:                                # nodes that have one type of attributes e.g., disease
                    param.requires_grad = False
                    
                    
                self.feat_all[k] = param
                self.fc_all[k] = nn.Sequential(
                        nn.Linear(v1.shape[1], d_model, device=device)
                        # nn.ReLU(),
                        # nn.Dropout(0.2),
                        # nn.Linear(3*d_model, 2*d_model, device=device),
                        # nn.ReLU(),
                        # nn.Linear(2*d_model, d_model, device=device)
                    )


        self.pos_emb = pos_emb # positional embedding
        if pos_emb == 'fixed':
            self.pos_encoding = self.create_positional_encoding(seq_len, d_model).to(device)    # Positional encoding (computed dynamically)
        elif pos_emb == 'learnable':
            self.pos_embedding = nn.Embedding(seq_len, d_model).to(device)
        
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model).to(device), 
        #     num_layers = N_encoder_layers
        # )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, activation='gelu').to(device),
            num_layers = N_encoder_layers
            )

        
        self.fc = nn.Linear(d_model, vocab_size+1).to(device)  # Output layer: Predict all vocab including [MASK]
        
        # binary classification layer 
        self.classifire = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # considering only head and tail for LP unless it should be d_model*3
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
        # x = torch.concat([x[:,0,:], x[:,1,:], x[:,2,:]], dim=1)
        # x = x[:, :3, :]              # shape: [batch, 3, hidden_dim]
        x = x[:, 0:3:2, :]              # shape: [batch, 2, hidden_dim]    # only considering head and tail for LP
        x = x.reshape(x.size(0), -1) # shape: [batch, 2 * hidden_dim]
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
            # if not matching_keys:
            #     print(f"No matching keys found for token: {t}")
            # else:
            #     print(f"Token {t} matched keys: {matching_keys}")
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
        # sorted_entity_indices = entity_indices[sorted_order]
        sorted_W = W[sorted_order]

        # for debugging
        # print("tokens.shape:", tokens.shape)
        # print("tokens.min():", tokens.min().item(), "tokens.max():", tokens.max().item())
        # print("embedding table size:", W.shape[0])
        # assert tokens.min() >= 0, "🚨 Token index is negative!"
        # assert tokens.max() < W.shape[0], "🚨 Token index exceeds embedding table size!"

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
        elif self.pos_emb == 'learnable':
            positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
            x_pos_emb = self.pos_embedding(positions)
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
        elif mask[i].sum() == data.shape[1]:                           ########## this was previously if
            mask[i, 0] = False
    masked_data[mask] = mask_token_id  # assign number of mask_token_id to the masks
    return masked_data, mask


def get_probs(context_idx, model, mask_token, seq_len, query=None, late_softmax=False):

    len_context = len(context_idx)
    assert len_context < seq_len
    
    context_idx = context_idx + [mask_token] * (seq_len - len_context)   # extend context with mask tokens
    context_idx = torch.tensor(context_idx).reshape(1, -1).to(model.device)
    out = model(context_idx)[0, len_context, :]                          # get the first masked token
    if not late_softmax:
        out = F.softmax(out, dim=0)
    if query != None:
        out = out[query]
    if late_softmax:
        out = F.softmax(out, dim=0)
    return out

##########################

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


########################################################################################################
class Classifier(nn.Module):
    def __init__(self, d_model, d_classifier, device):
        super(Classifier, self).__init__()
        
        # binary classification layer 
        layers = []
        input_dim = d_model * 2  # assuming head and tail concatenation
        for output_dim in d_classifier:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, 1))  # Final output layer
        self.classifier = nn.Sequential(*layers).to(device)

    def forward(self, x):
        # x = x[:, :3, :]              # shape: [batch, 3, hidden_dim]
        x = x[:, 0:3:2, :]              # shape: [batch, 2, hidden_dim]    # only considering head and tail for LP
        x = x.reshape(x.size(0), -1) # shape: [batch, 2 * hidden_dim]
        x = self.classifier(x)
        return x








