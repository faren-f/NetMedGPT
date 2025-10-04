# similarity calculation
import numpy as np
import torch
import torch.nn.functional as F 
from sklearn.metrics.pairwise import cosine_similarity


def sim(edge_label_index, model, device, mask_token, seq_len, batch_size, method='cosine', emb_select='raw'):
    model.eval()
    similarities = []

    # Split into batches
    for i in range(0, edge_label_index.shape[1], batch_size):
        # Get batch of node pairs
        batch_node1 = edge_label_index[0, i:i+batch_size]
        batch_edge = edge_label_index[1, i:i+batch_size]
        batch_node2 = edge_label_index[2, i:i+batch_size]

        # Create input tensor for the batch
        paths = torch.full((len(batch_node1), seq_len), mask_token, dtype=torch.long, device=device) # fill a matrix with dim (len(batch_node1), seq_len) with mask_token
        paths[:, 0] = batch_node1.to(device)
        paths[:, 1] = batch_edge.to(device)

        with torch.no_grad():
            out = model(paths)  # Shape: [B, seq_len, vocab_size+1]
            logits = out[:, 2, :]  # Extract predictions for position 1 (after the starting token)

            # separet edge type
            probs = F.softmax(logits, dim=1)  # Softmax across vocab dimension

            # Extract the predicted probability for each node2
            prob = probs[torch.arange(len(batch_node2)), batch_node2]
            similarities.append(prob)

    return torch.cat(similarities).cpu().numpy()


def edge2path(edge_label_index, device, mask_token, seq_len):
    
    batch_node1 = edge_label_index[:, 0]
    batch_edge = edge_label_index[:, 1]
    batch_node2 = edge_label_index[:, 2]
    paths = torch.full((len(batch_node1), seq_len), mask_token, dtype=torch.long, device=device)
    paths[:, 0] = batch_node1.to(device)
    paths[:, 1] = batch_edge.to(device)
    paths[:, 2] = batch_node2.to(device)
    return paths


def node_level_eval(edge_label_index, edge_label, all_ids_node2, model, device, mask_token, seq_len, k=100):
    model.eval()

    edge_label_index = edge_label_index[:, edge_label == 1]   # choose only positive samples
    # edge_type_token = edge_label_index[1].unique()   # extract relation tokens
    row_relation = edge_label_index[1]
    edge_type_token = torch.unique_consecutive(row_relation)


    hit_k_list = []
    precision_k_list = []

    # loop over each relation
    for e in edge_type_token:
        edge_label_index_e = edge_label_index[:, edge_label_index[1] == e] # choose edge index of relation e
        disease_unique = edge_label_index_e[0].unique()
        # drug_unique = edge_label_index_e[2].unique().to(device)
        drug_unique = all_ids_node2[e.item()].to(device)
    
        paths = torch.full((len(disease_unique), seq_len), mask_token, dtype=torch.long, device=device) # fill a matrix with dim (len(batch_node1), seq_len) with mask_token
        paths[:, 0] = disease_unique.to(device)
        paths[:, 1] = e.to(device)

        with torch.no_grad():
            out = model(paths)     # Shape: [disease_unique, seq_len, vocab_size+1]
            logits = out[:, 2, :]  # Extract predictions for position 2: for all the diseases only get the logits at the thrid position [len(disease_unique), vocab_size+1]
            logits = logits[:, drug_unique] #for all the diseases only get the logits of drugs that are in the edge index in the relation e, drug_unique [disease_unique, len(drug_unique)]
            # probs = F.softmax(logits, dim=1)  # Softmax across vocab dimension

        if len(disease_unique) != logits.shape[0]:
            print('Err')

        Hit_K = []
        Precision_K = []
        for i in range(logits.shape[0]):
            d = disease_unique[i]
            l = logits[i]
            top_k_drugs = drug_unique[torch.topk(l, k=int(k)).indices]
            drugs_pos = edge_label_index_e[2, edge_label_index_e[0] == d]

            if len(drugs_pos) > 0:
                hits_k = torch.isin(drugs_pos.to(device), top_k_drugs).sum() / len(drugs_pos) 
                precision_k = torch.isin(drugs_pos.to(device), top_k_drugs).sum() / k   

                Hit_K.append(hits_k.cpu().item())
                Precision_K.append(precision_k.cpu().item())

        # hit_k_dict[e.item()] = np.mean(Hit_K)
        hit_k_list.append(np.mean(Hit_K))
        precision_k_list.append(np.mean(Precision_K))


    return hit_k_list, precision_k_list  #, paths, disease_unique, top_k_drugs



def node_level_eval_old(edge_label_index, edge_label, all_ids_node2, model, device, mask_token, seq_len, k=100):
    model.eval()

    edge_label_index = edge_label_index[:, edge_label == 1]   # choose only positive samples
    # edge_type_token = edge_label_index[1].unique()   # extract relation tokens
    row_relation = edge_label_index[1]
    edge_type_token = torch.unique_consecutive(row_relation)


    hit_k_list = []
    precision_k_list = []

    # loop over each relation
    for e in edge_type_token:
        edge_label_index_e = edge_label_index[:, edge_label_index[1] == e] # choose edge index of relation e
        disease_unique = edge_label_index_e[0].unique()
        # drug_unique = edge_label_index_e[2].unique().to(device)
        drug_unique = all_ids_node2.to(device)

    
        paths = torch.full((len(disease_unique), seq_len), mask_token, dtype=torch.long, device=device) # fill a matrix with dim (len(batch_node1), seq_len) with mask_token
        paths[:, 0] = disease_unique.to(device)
        paths[:, 1] = e.to(device)

        with torch.no_grad():
            out = model(paths)     # Shape: [disease_unique, seq_len, vocab_size+1]
            logits = out[:, 2, :]  # Extract predictions for position 2: for all the diseases only get the logits at the thrid position [len(disease_unique), vocab_size+1]
            logits = logits[:, drug_unique] #for all the diseases only get the logits of drugs that are in the edge index in the relation e, drug_unique [disease_unique, len(drug_unique)]
            # probs = F.softmax(logits, dim=1)  # Softmax across vocab dimension

        if len(disease_unique) != logits.shape[0]:
            print('Err')

        Hit_K = []
        Precision_K = []
        for i in range(logits.shape[0]):
            d = disease_unique[i]
            l = logits[i]
            top_k_drugs = drug_unique[torch.topk(l, k=int(k)).indices]    
            drugs_pos = edge_label_index_e[2, edge_label_index_e[0] == d]

            if len(drugs_pos) > 0:
                hits_k = torch.isin(drugs_pos.to(device), top_k_drugs).sum() / len(drugs_pos) 
                precision_k = torch.isin(drugs_pos.to(device), top_k_drugs).sum() / k   

                Hit_K.append(hits_k.cpu().item())
                Precision_K.append(precision_k.cpu().item())

        # hit_k_dict[e.item()] = np.mean(Hit_K)
        hit_k_list.append(np.mean(Hit_K))
        precision_k_list.append(np.mean(Precision_K))


    return hit_k_list, precision_k_list
    

# not compelete
# def sim_finetune_v2(edge_label_index, model, device, mask_token, seq_len, batch_size, method='cosine', emb_select='raw'):
#     model.eval()
#     similarities = []

#     # Split into batches
#     for i in range(0, edge_label_index.shape[1], batch_size):
#         # Get batch of node pairs
#         batch_node1 = edge_label_index[0, i:i+batch_size]
#         batch_edge = edge_label_index[1, i:i+batch_size]
#         batch_node2 = edge_label_index[2, i:i+batch_size]

#         # Create input tensor for the batch
#         paths = torch.full((len(batch_node1), seq_len), mask_token, dtype=torch.long, device=device) # fill a matrix with dim (len(batch_node1), seq_len) with mask_token
#         paths[:, 0] = batch_node1.to(device)
#         paths[:, 1] = batch_edge.to(device)

#         with torch.no_grad():
#             out = model(paths)  # Shape: [B, seq_len, vocab_size+1]
#             logits = out[:, 2, :]  # Extract predictions for position 1 (after the starting token)

#             # Extract the predicted probability for each node2 (torch advance indexing)
#             logit = logits[torch.arange(len(batch_node2)), batch_node2]

#             similarities.append(logit)

#     return torch.cat(similarities).cpu().numpy()



#### this function is used for clinical trial analysis, when we find logits of drugs from the whole tokens 
def sim_with_logit(edge_label_index, model, device, mask_token, seq_len, batch_size, method='cosine', emb_select='raw'):
    model.eval()
    similarities = []

    # Split into batches
    for i in range(0, edge_label_index.shape[1], batch_size):
        # Get batch of node pairs
        batch_node1 = edge_label_index[0, i:i+batch_size]
        batch_edge = edge_label_index[1, i:i+batch_size]
        batch_node2 = edge_label_index[2, i:i+batch_size]

        # Create input tensor for the batch
        paths = torch.full((len(batch_node1), seq_len), mask_token, dtype=torch.long, device=device) # fill a matrix with dim (len(batch_node1), seq_len) with mask_token
        paths[:, 0] = batch_node1.to(device)
        paths[:, 1] = batch_edge.to(device)

        with torch.no_grad():
            out = model(paths)  # Shape: [B, seq_len, vocab_size+1]
            print("out.shape:", tuple(out.shape), "dtype:", out.dtype, "device:", out.device)

            logits = out[:, 2, :]  # Extract predictions for position 1 (after the starting token)
            print("logits_allpos.shape:", tuple(logits.shape))


            # Extract the predicted probability for each node2
            logits = logits[torch.arange(len(batch_node2)), batch_node2]
            similarities.append(logits)

    return torch.cat(similarities).cpu().numpy()



#### this function ca be used for clinical trial analysis, when we find rank of drugs in the logits 
def sim_with_rank(edge_label_index, model, device, mask_token, seq_len, batch_size, drug_ind, method='cosine', emb_select='raw'):
    model.eval()
    similarities = []

    # Split into batches
    for i in range(0, edge_label_index.shape[1], batch_size):
        # Get batch of node pairs
        batch_node1 = edge_label_index[0, i:i+batch_size]
        batch_edge = edge_label_index[1, i:i+batch_size]
        batch_node2 = edge_label_index[2, i:i+batch_size]

        # Create input tensor for the batch
        paths = torch.full((len(batch_node1), seq_len), mask_token, dtype=torch.long, device=device) # fill a matrix with dim (len(batch_node1), seq_len) with mask_token
        paths[:, 0] = batch_node1.to(device)
        paths[:, 1] = batch_edge.to(device)

        with torch.no_grad():
            out = model(paths)  # Shape: [B, seq_len, vocab_size+1]
            logits = out[:, 2, :]  # Extract predictions for position 1 (after the starting token)

            logits = logits.cpu()
            # Extract the rank for each node2
            mask = torch.full_like(logits, float('-inf'))
            mask[:, drug_ind] = logits[:, drug_ind]
            logits = mask
            sorted_indices = torch.argsort(logits, dim=1, descending=True)
            ranks = (sorted_indices == batch_node2.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]

            # logits = logits[torch.arange(len(batch_node2)), batch_node2]
            similarities.append(ranks)

    return torch.cat(similarities).cpu().numpy()




@torch.inference_mode()
def sim_with_logit_efficient_memory(edge_label_index, model, device, mask_token, seq_len, batch_size, method='cosine', emb_select='raw'):
    model.eval()
    similarities = []
    W, b = model.fc.weight, model.fc.bias  # [V,d], [V] or None

    # Split into batches
    for i in range(0, edge_label_index.shape[1], batch_size):
        # Get batch of node pairs
        batch_node1 = edge_label_index[0, i:i+batch_size]
        batch_edge  = edge_label_index[1, i:i+batch_size]
        batch_node2 = edge_label_index[2, i:i+batch_size]

        # Create input tensor for the batch
        paths = torch.full((len(batch_node1), seq_len), mask_token, dtype=torch.long, device=device)
        paths[:, 0] = batch_node1.to(device)
        paths[:, 1] = batch_edge.to(device)

        # ---- only change: skip model(paths) to avoid fc ----
        h = model.compute_embedding(paths) + model.get_x_pos_emb(paths)   # [B,S,d]
        h = model.transformer(h.permute(1,0,2)).permute(1,0,2)            # [B,S,d]
        h = h[:, 2, :]                                                    # [B,d]

        # project only onto the needed targets
        W_sel = W[batch_node2.to(W.device), :]                            # [B,d]
        logits = (h * W_sel.to(h.device)).sum(dim=1)                      # [B]
        if b is not None:
            logits = logits + b[batch_node2.to(b.device)].to(h.device)

        similarities.append(logits.detach().cpu())

        del paths, h, W_sel, logits
        torch.cuda.empty_cache()

    return torch.cat(similarities).numpy()



















