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
def print_eval_results(epoch, loss, eval_results, relation_names=None):
    print(f"Epoch {epoch}, Loss: {loss:.4f}, link pred results:\n")
    for i, df in enumerate(eval_results):
        name = relation_names[i] if relation_names else f"Relation {i+1}"
        print(f"--- {name} ---")
        print(df.to_string(index=False))
        print()  # Add space between tables


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

    
########################################################################
# def link_pred_val(pred_test : torch.tensor, target_test : torch.tensor, edge_type_flag_test: torch.tensor, param, key = ['auc','auprc', 'prc@5','hits@5','prc@50', 'hits@50', 'prc@100', 'hits@100', 'mrr']):
    
#     pred_test_drug_adr = pred_test[edge_type_flag_test == 1]
#     pred_test_drug_gene = pred_test[edge_type_flag_test == 2]
#     pred_test_disease_dp = pred_test[edge_type_flag_test == 3]
#     pred_test_gene_disease = pred_test[edge_type_flag_test == 4]
#     pred_test_drug_disease = pred_test[edge_type_flag_test == 5]

    
#     target_test_drug_adr = target_test[edge_type_flag_test == 1]
#     target_test_drug_gene = target_test[edge_type_flag_test == 2]
#     target_test_disease_dp = target_test[edge_type_flag_test == 3]
#     target_test_gene_disease = target_test[edge_type_flag_test == 4]
#     target_test_drug_disease = target_test[edge_type_flag_test == 5]


#     lp_drug_adr = Evaluation(target_test_drug_adr, pred_test_drug_adr).eval(param["evaluation"]["top_k"])
#     lp_gene_drug = Evaluation(target_test_drug_gene, pred_test_drug_gene).eval(param["evaluation"]["top_k"])
#     lp_disease_dp = Evaluation(target_test_disease_dp, pred_test_disease_dp).eval(param["evaluation"]["top_k"])
#     lp_gene_disease = Evaluation(target_test_gene_disease, pred_test_gene_disease).eval(param["evaluation"]["top_k"])
#     lp_drug_disease = Evaluation(target_test_drug_disease, pred_test_drug_disease).eval(param["evaluation"]["top_k"])


    
#     return lp_drug_adr[key], lp_gene_drug[key], lp_disease_dp[key], lp_gene_disease[key], lp_drug_disease[key]
#     #return lp_drug_adr[key].item(), lp_gene_drug[key].item(), lp_disease_dp[key].item(), lp_gene_disease[key].item(), lp_drug_disease[key].item() # when key is one item


##################################

def calculate_metrics(preds, targets):
    """
    Calculate sensitivity, specificity, and accuracy.
    """
    # True Positives (TP)
    TP = ((preds == 1) & (targets == 1)).float().sum()
    # True Negatives (TN)
    TN = ((preds == 0) & (targets == 0)).float().sum()
    # False Positives (FP)
    FP = ((preds == 1) & (targets == 0)).float().sum()
    # False Negatives (FN)
    FN = ((preds == 0) & (targets == 1)).float().sum()
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else torch.tensor(0.0)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else torch.tensor(0.0)
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else torch.tensor(0.0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else torch.tensor(0.0)
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else torch.tensor(0.0)

    
    return sensitivity, specificity, accuracy, precision, F1



