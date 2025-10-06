import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score



def link_pred_val(pred_test: torch.Tensor, target_test: torch.Tensor, edge_type_flag_test: torch.Tensor, param, edge_type, key=['auc', 'auprc','prc@5', 'hits@5', 'prc@100','hits@100', 'mrr']):
      
    lp_i = pd.DataFrame(columns=key)
    for i in edge_type:
        mask = torch.isin(edge_type_flag_test, torch.tensor(i, device=edge_type_flag_test.device))
        pred_test_i = pred_test[mask]
        target_test_i = target_test[mask]
        lp_i = pd.concat([lp_i, Evaluation(target_test_i, pred_test_i).eval(param["evaluation"]["top_k"])])
        # val_result = Evaluation(target_test_i, pred_test_i).eval(param["evaluation"]["top_k"])
    return lp_i


#################################################################################

class Evaluation():
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_pos = y_pred[y_true == 1]
        self.y_pred_neg = y_pred[y_true == 0]

    def eval(self, k_list):

        df = pd.DataFrame(columns = ['prc@' + str(k) for k in k_list] + 
                                    ['hits@' + str(k) for k in k_list] +
                                    ['recall@' + str(k) for k in k_list] +
                                    ['mrr', 'auc', 'auprc'])

        
        Precision_k_list = []
        Hits_k_list = []
        Recall_k_list = []
        for k in k_list:
            # Precision_k
            Precision_k_list.append(self.precision_k(k))
            Hits_k_list.append(self.hits_k(k))
            Recall_k_list.append(self.recall_k(k))


        # MRR_list = self.mrr()
        MRR = [self.mrr()]
        AUC = [self.auc()]
        AUPRC = [self.auprc()]
        df.loc[0] = Precision_k_list + Hits_k_list + Recall_k_list+ MRR + AUC + AUPRC

        return df


    def precision_k(self, k):
        """
        the fraction of true links that appear in the first 𝑘 link of the sorted rank list.
        
        y_true: A tensor of ground truth (0 and 1).
        y_pred: A tensor of logits.
        k: Number of top elements to look at for computing precision.
        """
        # Get indices of the predictions with the highest scores
        if k > len(self.y_pred):
            k = len(self.y_pred)
        
        topk_indices = torch.topk(self.y_pred, k).indices
        
        # Calculate precision
        value = self.y_true[topk_indices].float().mean()
        return value.item()
    
    
    def hits_k(self, k):
        
        if k > len(self.y_pred_neg):
            return(1)
            
        # find the kth score in the negative predictions as the threshold
        kth_score_in_negative_edges = torch.topk(self.y_pred_neg, k=k)[0][-1]
        # Count the number of positive predictions > the threshold the / len(y_pred_pos)
        hitsK = float(torch.sum(self.y_pred_pos > kth_score_in_negative_edges).cpu()) / len(self.y_pred_pos)
        return hitsK

    def recall_k(self, k):
        if k > len(self.y_pred):
            k = len(self.y_pred)
    
        topk_indices = torch.topk(self.y_pred, k).indices
        true_positives = self.y_true[topk_indices].sum().item()
        total_positives = self.y_true.sum().item()

        
        if total_positives == 0:
            return 0.0  # or np.nan depending on your setup
        
        recall = true_positives / total_positives
        return recall
    
    
    def mrr2(self):
        '''
            compute mrr
            y_pred_neg is an array with shape (batch size, num_entities_neg).
            y_pred_pos is an array with shape (batch size, num_entities_pos)
        '''
        
        # calculate ranks
        y_pred_pos = self.y_pred_pos.view(-1, 1)
        # optimistic rank: "how many negatives have at least the positive score?"
        # ~> the positive is ranked first among those with equal score
        optimistic_rank = (self.y_pred_neg >= y_pred_pos).sum(dim=1)
        # pessimistic rank: "how many negatives have a larger score than the positive?"
        # ~> the positive is ranked last among those with equal score
        pessimistic_rank = (self.y_pred_neg > y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    
        hits1_list = (ranking_list <= 1).to(torch.float)
        hits3_list = (ranking_list <= 3).to(torch.float)
        hits10_list = (ranking_list <= 10).to(torch.float)
        hits20_list = (ranking_list <= 20).to(torch.float)
        hits50_list = (ranking_list <= 50).to(torch.float)
        hits100_list = (ranking_list <= 100).to(torch.float)
        mrr_list = 1./ranking_list.to(torch.float)
    
        return [hits1_list.mean().item(),
                hits3_list.mean().item(),
                hits10_list.mean().item(),
                hits20_list.mean().item(),
                hits50_list.mean().item(),
                hits100_list.mean().item(),
                mrr_list.mean().item()]
    
    def mrr(self):
        """
        Calculate the Mean Reciprocal Rank (MRR) for link prediction.
    
        y_true: A tensor of ground truth (0 and 1).
        y_pred: A tensor of logits.
        """
    
        # Sort y_pred in descending order and get the indices
        sorted_indices = torch.argsort(self.y_pred, descending=True) +1
    
        # Calculate the mean reciprocal rank
        mrr = (1 / sorted_indices[self.y_true == 1]).mean()
        return mrr.item()

    # def auc(self):
    #     # try:
    #     auc_score = roc_auc_score(self.y_true.detach().cpu().numpy(), self.y_pred.detach().cpu().numpy())
    #     # except ValueError:  # Handle case with only one class present in y_true
    #     #    auc_score = float('nan')
    #     return auc_score

     # for debugging

    def auc(self):
        y_true_np = self.y_true.detach().cpu().numpy()
        y_pred_np = self.y_pred.detach().cpu().numpy()
        try:
            auc_score = roc_auc_score(y_true_np, y_pred_np)
        except ValueError as e:
            print(f"❗AUC computation failed: {e}")
            print(f"   y_true: {np.unique(y_true_np, return_counts=True)}")
            return float('nan')
        return auc_score

    
    def auprc(self):
        """
        Compute Area Under the Precision-Recall Curve (AUPRC)
        """
        y_true_ = self.y_true.detach().cpu().numpy()
        y_pred_ = self.y_pred.detach().cpu().numpy()
        return average_precision_score(y_true_, y_pred_)

#############
# sim
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

