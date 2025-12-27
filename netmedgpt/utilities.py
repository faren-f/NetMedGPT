import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import torch.nn.functional as F 
from sklearn.metrics.pairwise import cosine_similarity

def link_pred_val(pred_test: torch.Tensor, target_test: torch.Tensor, edge_type_flag_test: torch.Tensor, param, edge_type, key=['auc', 'auprc']):
      
    lp_i = pd.DataFrame(columns=key)
    for i in edge_type:
        mask = torch.isin(edge_type_flag_test, torch.tensor(i, device=edge_type_flag_test.device))
        pred_test_i = pred_test[mask]
        target_test_i = target_test[mask]
        df_i = Evaluation(target_test_i, pred_test_i).eval(param["evaluation"]["top_k"])
        if not df_i.empty and not df_i.isna().all(axis=None):
            lp_i = pd.concat([lp_i, df_i], ignore_index=True)
    return lp_i


class Evaluation():
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_pos = y_pred[y_true == 1]
        self.y_pred_neg = y_pred[y_true == 0]

    def eval(self, k_list):
        df = pd.DataFrame(columns = ['auc', 'auprc'])
        AUC = [self.auc()]
        AUPRC = [self.auprc()]
        df.loc[0] = AUC + AUPRC
        return df    

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
# similarity calculation
def sim(edge_label_index, model, device, mask_token, seq_len, batch_size, method='cosine', emb_select='raw'):
    model.eval()
    similarities = []

    # Split into batches
    for i in range(0, edge_label_index.shape[1], batch_size):
        batch_node1 = edge_label_index[0, i:i+batch_size]
        batch_edge = edge_label_index[1, i:i+batch_size]
        batch_node2 = edge_label_index[2, i:i+batch_size]

        # Create input tensor for the batch
        paths = torch.full((len(batch_node1), seq_len), mask_token, dtype=torch.long, device=device) 
        paths[:, 0] = batch_node1.to(device)
        paths[:, 1] = batch_edge.to(device)

        with torch.no_grad():
            out = model(paths) 
            logits = out[:, 2, :]  

            # separet edge type
            probs = F.softmax(logits, dim=1)
            prob = probs[torch.arange(len(batch_node2)), batch_node2]
            similarities.append(prob)
    return torch.cat(similarities).cpu().numpy()

def node_level_eval(edge_label_index, edge_label, all_ids_node2, model, device, mask_token, seq_len, k=100):
    model.eval()

    edge_label_index = edge_label_index[:, edge_label == 1] 
    row_relation = edge_label_index[1]
    edge_type_token = torch.unique_consecutive(row_relation)

    recall_k_list = []
    precision_k_list = []

    # loop over each relation
    for e in edge_type_token:
        edge_label_index_e = edge_label_index[:, edge_label_index[1] == e] # choose edge index of relation e
        disease_unique = edge_label_index_e[0].unique()
        drug_unique = all_ids_node2[e.item()].to(device)
    
        paths = torch.full((len(disease_unique), seq_len), mask_token, dtype=torch.long, device=device) 
        paths[:, 0] = disease_unique.to(device)
        paths[:, 1] = e.to(device)

        with torch.no_grad():
            out = model(paths)     
            logits = out[:, 2, :]  
            logits = logits[:, drug_unique] 

        if len(disease_unique) != logits.shape[0]:
            print('Err')

        Recall_K = []
        Precision_K = []
        for i in range(logits.shape[0]):
            d = disease_unique[i]
            l = logits[i]
            top_k_drugs = drug_unique[torch.topk(l, k=int(k)).indices]
            drugs_pos = edge_label_index_e[2, edge_label_index_e[0] == d]

            if len(drugs_pos) > 0:
                recalls_k = torch.isin(drugs_pos.to(device), top_k_drugs).sum() / len(drugs_pos) 
                precision_k = torch.isin(drugs_pos.to(device), top_k_drugs).sum() / k   

                Recall_K.append(recalls_k.cpu().item())
                Precision_K.append(precision_k.cpu().item())

        recall_k_list.append(np.mean(Recall_K))
        precision_k_list.append(np.mean(Precision_K))
    return recall_k_list, precision_k_list

