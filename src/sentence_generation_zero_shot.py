import os
import json
import random

import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from tqdm import tqdm


def sentence_generation(edge_index, nodes, queried_node_types, queried_edge_types, test_ratio, val_ratio, relation_node_types, param, seed):
    queried_dict = {i:relation_node_types[i] for i in queried_edge_types}

    ## Train-val-test split of edge list of queried types
    val_data_edge_label_index = torch.tensor([], dtype=torch.long)
    val_data_edge_label = torch.tensor([], dtype=torch.long)
    edge_type_flag_val = torch.tensor([], dtype=torch.long)
    
    test_data_edge_label_index = torch.tensor([], dtype=torch.long)
    test_data_edge_label = torch.tensor([], dtype=torch.long)
    edge_type_flag_test = torch.tensor([], dtype=torch.long)
    
    to_remove_pretrain = []
    for r, (i, (node_source1, node_source2)) in enumerate(queried_dict.items()):
        set_seed(seed)  
        print(f"relation: {r}")
        # Make one-sided edges (currently valid only for disease-drug ralations)
        edges_i = edge_index.loc[
                (edge_index['relation'] == i) & (edge_index['node_types'] == queried_node_types[r]), 
                ["x_index", "y_index", "xy"]].copy()
    
        edges_i = edges_i.drop_duplicates()
        edges_shuffled = edges_i.sample(frac=1, random_state = seed).reset_index(drop=True) 
    
        # all disease nodes in relation_i
        nodes_i = edges_shuffled['x_index'].unique()  # all diseases nodes  [one sided 'disease|drug']
        # print(len(nodes_i))
    
        #################################################################
        # seperate test nodes
        num_holdout_test = int(test_ratio * len(nodes_i))
        test_nodes = set(random.sample(list(nodes_i), num_holdout_test))
        # print(len(test_nodes))
        
        # Split edges into test and trainval:
        test_edges_one_side = edges_shuffled[edges_shuffled['x_index'].isin(test_nodes)]
        # all drugs that are connected to test nodes
        all_drug_ids = edges_i.loc[:,'y_index'].unique()
        negative_edges_test = negative_samples(test_edges_one_side, edge_index, all_drug_ids, seed)
        # print(test_edges_one_side.shape)
        # print(len(negative_edges_test))
    
        trainval_edges = edges_shuffled[~edges_shuffled['x_index'].isin(test_nodes)]
        trainval_shuffled = trainval_edges.sample(frac=1, random_state=seed)
        nodes_i = trainval_shuffled['x_index'].unique()  # which are diseases in indication one sided 'disease|drug'
        num_holdout_val = int(val_ratio * len(nodes_i))
        val_nodes = set(random.sample(list(nodes_i), num_holdout_val))
        val_edges_one_side = trainval_shuffled[trainval_shuffled['x_index'].isin(val_nodes)]        
        negative_edges_val = negative_samples(val_edges_one_side, edge_index, all_drug_ids, seed)
    
        positive_edges_val = list(zip(val_edges_one_side["x_index"], val_edges_one_side["y_index"]))
        positive_edges_test = list(zip(test_edges_one_side["x_index"], test_edges_one_side["y_index"]))
                                                            
        ## Convert to torch
        positive_edges_val = to_edge_tensor(positive_edges_val)
        positive_edges_test = to_edge_tensor(positive_edges_test)
    
        negative_edges_val = to_edge_tensor(negative_edges_val)
        negative_edges_test = to_edge_tensor(negative_edges_test)
    
        val_data_edge_label_index_i = torch.cat((positive_edges_val, negative_edges_val), dim = 1)
        test_data_edge_label_index_i = torch.cat((positive_edges_test, negative_edges_test), dim = 1)
    
        
        val_data_edge_label_i = torch.cat((torch.ones(positive_edges_val.shape[1], dtype=torch.long), 
                                            torch.zeros(negative_edges_val.shape[1], dtype=torch.long)), dim = 0)
        test_data_edge_label_i = torch.cat((torch.ones(positive_edges_test.shape[1], dtype=torch.long), 
                                            torch.zeros(negative_edges_test.shape[1], dtype=torch.long)), dim = 0)
    
        edge_type_flag_val_i = (r+1) * torch.ones(val_data_edge_label_i.shape, dtype=torch.long)
        edge_type_flag_test_i = (r+1) * torch.ones(test_data_edge_label_i.shape, dtype=torch.long)
    
        val_data_edge_label_index = torch.cat((val_data_edge_label_index, val_data_edge_label_index_i), dim = 1)
        test_data_edge_label_index = torch.cat((test_data_edge_label_index, test_data_edge_label_index_i), dim = 1)
    
        val_data_edge_label = torch.cat((val_data_edge_label, val_data_edge_label_i), dim = 0)
        test_data_edge_label = torch.cat((test_data_edge_label, test_data_edge_label_i), dim = 0)
    
        edge_type_flag_val = torch.cat((edge_type_flag_val, edge_type_flag_val_i), dim = 0)
        edge_type_flag_test = torch.cat((edge_type_flag_test, edge_type_flag_test_i), dim = 0)
    
        ## Remove test edges (positive_edges) from the train edges
        reversed_positive_edges_val = positive_edges_val.flip(0)
        reversed_positive_edges_test = positive_edges_test.flip(0)
    
        positive_edges_2sided_val = torch.cat([positive_edges_val, reversed_positive_edges_val], dim=1)
        positive_edges_2sided_test = torch.cat([positive_edges_test, reversed_positive_edges_test], dim=1)
        positive_edges_val_test = torch.cat([positive_edges_2sided_val, positive_edges_2sided_test], dim=1)
    
        to_remove_pretrain.extend([f'{x}|{y}' for x,y in positive_edges_val_test.T])

        ####################################
        #### debugging: check if the test nodes are in the trainval nodes
        nodes_to_check = positive_edges_test[0].tolist() # get all disease nodes
        df_nodes = set(trainval_edges['x_index']).union(set(trainval_edges['y_index']))     # Get all unique nodes in the trainval_edges
        # Check if any node is present in trainval
        any_common = any(node in df_nodes for node in nodes_to_check)
        if any_common:
            print("❌ There is overlap between test and trainval nodes.")
        else:
            print("✅ No overlap found between test and trainval nodes.")
        ####################################
        # again check if the test nodes are in the trainval nodes
        test_diseases = set(test_edges_one_side["x_index"])
        trainval_diseases = set(trainval_edges["x_index"])
        
        intersect = test_diseases.intersection(trainval_diseases)
        if intersect:
            print(f"Inductive violation: {len(intersect)} test diseases are in training!")
        else:
            print("Inductive isolation successful. No test diseases in train/val.")
    
    
    train_edge_index_df = edge_index.loc[~edge_index['xy'].isin(to_remove_pretrain), ['x_index', 'y_index']]
    train_data_edge_index = train_edge_index_df.to_numpy().T
    train_data_edge_index_pretrain = torch.tensor(train_data_edge_index, dtype = torch.long)
    
    ########debugging
    # check if the test edges are in the walks
    edges_A = set(zip(test_data_edge_label_index[0].tolist(), test_data_edge_label_index[1].tolist()))
    edges_B = set(zip(train_data_edge_index_pretrain[0].tolist(), train_data_edge_index_pretrain[1].tolist()))
    overlap = edges_A.intersection(edges_B)
    
    print(f"Number of overlapping edges: {len(overlap)}")
    print("Overlapping edges:", overlap)
    
    #######
    edges_A = set(zip(positive_edges_2sided_test[0].tolist(), positive_edges_2sided_test[1].tolist()))
    edges_B = set(zip(train_data_edge_index_pretrain[0].tolist(), train_data_edge_index_pretrain[1].tolist()))
    
    # Find overlap
    overlap = edges_A.intersection(edges_B)
    
    print(f"Number of overlapping edges: {len(overlap)}")
    print("Overlapping edges:", overlap)
    
    
    ############
    set_test_edges = set(zip(test_data_edge_label_index[0].tolist(), test_data_edge_label_index[1].tolist()))
    set_in_walk = set(zip(train_edge_index_df['x_index'], train_edge_index_df['y_index']))
    
    overlap = set_test_edges.intersection(set_in_walk)
    
    # Print results
    print(f"Number of overlapping edges: {len(overlap)}")
    
    
    ################
    set_seed(seed)      
    model = Node2Vec(
        edge_index = train_data_edge_index_pretrain,
        embedding_dim=1,                # embedding_dim can be dummy if you're not training
        walk_length=param['node2vec']['walk_length'],
        context_size= param['node2vec']['walk_length'],
        walks_per_node=param['node2vec']['walks_per_node'],
        p=1.0, q=1.0,
        num_negative_samples=0,         # no negative sampling needed
        sparse=True)
    
    walk_loader = model.loader(batch_size=param['node2vec']['batch_size'], shuffle=False, num_workers = param['node2vec']['workers'])
    
    all_walks = []
    for pos_rw, _ in walk_loader:
        all_walks.extend(pos_rw.tolist())  # only use the positive random walks
    
    total = len(all_walks)
    index = np.random.choice(np.arange(0, total), size=total, replace=False)
    all_walks = np.array(all_walks)
    all_walks_shuffled = all_walks[index]
    node2vec_walks = torch.tensor(all_walks_shuffled)

    ######insert relation
    walks_df = edge_index.loc[:,["x_index", "y_index", "z_index"]]
    
    # make a dict of edges and their relations
    edge_to_rel = {
        (x, y): z
        for x, y, z in zip(walks_df['x_index'], walks_df['y_index'], walks_df['z_index'])}

    # insert relation in walks
    walks = []
    for walk in tqdm(node2vec_walks):
        x = add_relations_to_walk(walk, edge_to_rel)
        if x is not None: 
            walks.append(x)
        
    walks_tensor = torch.tensor(walks, dtype=torch.long)
    
    # insert relation into val and test data
    test_relation_row_all = torch.empty((1, 0), dtype=torch.long)
    val_relation_row_all = torch.empty((1, 0), dtype=torch.long)
    for i in range(len(queried_edge_types)):
        relation_token_i = edge_index.loc[edge_index['relation'] == queried_edge_types[i], 'z_index'].unique()
        if len(relation_token_i) != 1:
            raise ValueError(f"Expected one unique relation token for relation {queried_edge_types[i]}, got {relation_token_i}")
        
        val_data_edge_label_index_i = val_data_edge_label_index[:,edge_type_flag_val == i+1]
        test_data_edge_label_index_i = test_data_edge_label_index[:, edge_type_flag_test == i+1]
    
        num_val_edges_i = val_data_edge_label_index_i.shape[1]
        num_test_edges_i = test_data_edge_label_index_i.shape[1]
    
        val_relation_row_i = torch.full((1, num_val_edges_i), relation_token_i.item(), dtype=val_data_edge_label_index_i.dtype)
        test_relation_row_i = torch.full((1, num_test_edges_i), relation_token_i.item(), dtype = test_data_edge_label_index_i.dtype)
    
        val_relation_row_all = torch.cat([val_relation_row_all, val_relation_row_i], dim=1)
        test_relation_row_all = torch.cat([test_relation_row_all, test_relation_row_i], dim=1)
    
    
    val_data_edge_label_index_with_relation = torch.cat([val_data_edge_label_index, val_relation_row_all], dim=0)
    val_data_edge_label_index_with_relation = val_data_edge_label_index_with_relation[[0,2,1],:]
    
    test_data_edge_label_index_with_relation = torch.cat([test_data_edge_label_index, test_relation_row_all], dim=0)
    test_data_edge_label_index_with_relation = test_data_edge_label_index_with_relation[[0,2,1],:]
    
    data = {
        'node2vec_walks': walks_tensor,
        'val_data_edge_label_index_with_relation': val_data_edge_label_index_with_relation,
        'val_data_edge_label': val_data_edge_label,
        'edge_type_flag_val': edge_type_flag_val,
        'test_data_edge_label_index_with_relation': test_data_edge_label_index_with_relation,
        'test_data_edge_label': test_data_edge_label,
        'edge_type_flag_test': edge_type_flag_test,
        'train_data_edge_index_pretrain':train_data_edge_index_pretrain
    }
    return data
 
##################################################
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def negative_samples(pos_edges, all_edges, all_drugs, seed=42):
    np.random.seed(seed)

    # Build a mapping of disease → known positive drugs
    disease_to_pos = pos_edges.groupby('x_index')['y_index'].apply(set).to_dict()
    # print("Total number of drugs:", len(all_drugs))
    # print("Most drugs connected to one disease:", max(len(v) for v in disease_to_pos.values()))
    
    
    # Set of all known disease-drug edges to avoid duplicates
    known_edges = set(zip(all_edges['x_index'], all_edges['y_index']))
    
    negative_edges = []
    
    for disease, pos_drugs in disease_to_pos.items():
        candidate_drugs = list(set(all_drugs) - pos_drugs)
        # print(f"\n Disease {disease}")
        # print(f"  #Positives: {len(pos_drugs)}")
        # print(f" Candidate negatives available: {len(candidate_drugs)}")
        
        if not candidate_drugs:
            print(f"Disease {disease} has no negative drug candidates! Skipping.")
            continue   
    
        num_negatives_needed = len(pos_drugs)
        selected_negatives = set()
        np.random.shuffle(candidate_drugs)  # optional: randomize order
        for drug in candidate_drugs:
            if (disease, drug) not in known_edges:
                selected_negatives.add((disease, drug))
            if len(selected_negatives) == num_negatives_needed:
                break

        # print(f" Selected negatives: {len(selected_negatives)}")

    
        # Add selected negatives to final list
        negative_edges.extend(selected_negatives)
    
        # print(f"Disease {disease}: +{len(pos_drugs)} / -{len(selected_negatives)}")

    return(negative_edges)


def to_edge_tensor(edge_list):
    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    else:
        return torch.tensor(edge_list, dtype=torch.long).T

def add_relations_to_walk(walk, edge_to_rel):
    sequence = []
    for i in range(len(walk) - 1):
        src, dst = walk[i].item(), walk[i + 1].item()
        rel = edge_to_rel.get((src, dst))
        if rel is None:
            return None  # skip this walk
        sequence.extend([src, rel])
    sequence.append(walk[-1].item())
    return sequence






