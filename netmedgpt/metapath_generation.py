import torch
import torch.nn.functional as F

def get_logits(walk, position, model):
    
    """
    For a given walk and position, outputs logits of all tokens for on that position
    """
    model.eval()
    outputs_list = []
    with torch.no_grad():
        output = model(walk)
        outputs_list.append(output)
        
    stacked_outputs = torch.stack(outputs_list)  # (rep, 1, seq_len, vocab_size)
    logits_at_mask = stacked_outputs[:, 0, position, :][0]  # (rep, vocab_size)
    return logits_at_mask

def get_top_k_tokens_from_logits(logits, k, node_edge_indecices):
    topk_logits, topk_indices = torch.topk(logits, k=k)
    topk_names = node_edge_indecices.set_index("node_index").loc[topk_indices.cpu().numpy()]["node_name"]
    return topk_names, topk_logits
    
def get_token_from_walk(walk, position, model, node_edge_indecices, k=5, T=1.0):
    logits = get_logits(walk, position, model)
    topk_tokens, topk_logits = get_top_k_tokens_from_logits(logits, k, node_edge_indecices)
    # print(topk_tokens)
    probs = F.softmax(topk_logits, dim=-1)

    # Apply temprature method
    scaled_logits = topk_logits / T
    probs_scaled = F.softmax(scaled_logits, dim=-1)

    # print(probs_scaled)
    sampled_index = torch.multinomial(probs_scaled, num_samples=1).item()
    token = topk_tokens.index[sampled_index]
    # print(token)
    prob_scaled = probs_scaled[sampled_index].item()
    prob = probs[sampled_index].item()

    # print(f"prob_scaled:{prob_scaled}"), print(f"prob:{prob}")
    return token, prob_scaled, prob

def metapath_generation(walk_input, n_metapath, mask_token, model, node_edge_indecices):
    """
    Given a walk_input, this function sequqntally performs next 'mask' token prediction.
    """
    k = 5
    L = len(walk_input[0])

    all_metapath = []
    all_probs_metapath = []
    all_probs_scaled_metapath = []
    for rep in range(n_metapath):
        walk = walk_input.clone()
        probs = torch.ones_like(walk).to(torch.double)
        probs_scaled = torch.ones_like(walk).to(torch.double)

        for pos in range(L):
            if walk[0, pos].item() == mask_token:
                topk_tokens, topk_probs_scaled, topk_probs = get_token_from_walk(walk, pos, model, node_edge_indecices, k=5, T=10)
                walk[0, pos] = topk_tokens 
                probs_scaled[0, pos] = topk_probs_scaled
                probs[0, pos] = topk_probs

        all_metapath.append(walk)
        all_probs_scaled_metapath.append(probs_scaled)
        all_probs_metapath.append(probs)

    return all_metapath, all_probs_scaled_metapath, all_probs_metapath 
