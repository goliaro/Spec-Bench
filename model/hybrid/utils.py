import copy
import random

# typing 
from typing import List, Tuple
import time
import torch

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers


def initialize_tree0(input_ids, model, past_key_values, logits_processor):
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor
    )

    #     if logits_processor is not None:
    #         logits = orig[:, -1]
    #         logits = logits_processor(None, logits)
    #         probabilities = torch.nn.functional.softmax(logits, dim=1)
    #         token = torch.multinomial(probabilities, 1)
    #     else:
    #         token = torch.argmax(orig[:, -1])
    #         token = token[None, None]
    #     input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    #     # Clone the output hidden states
    #
    #     draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head)
    #     if output_orig:
    #         return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, orig, hidden_states, token
    #     return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, hidden_states, token
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token

def construct_retrieve_indices(token_ids, parents, device):
    # Find all indices that are used as parents
    parent_set = set(parents)
    
    # Get all indices
    all_indices = set(range(len(token_ids)))
    
    # Leaves are indices that don't appear as parents
    leaf_indices = sorted(list(all_indices - parent_set))
    
    # For each leaf, trace path back to root
    paths = []
    for leaf_idx in leaf_indices:
        path = [leaf_idx]
        current = leaf_idx
        
        # Trace back to root
        while parents[current] != -1:  # Assuming -1 indicates the root
            current = parents[current]
            path.append(current)
        
        # Reverse to get path from root to leaf
        path.reverse()
        paths.append(path)
    
    # Find maximum depth
    max_depth = max(len(path) for path in paths)
    
    # Create retrieve_indices tensor padded with -1
    retrieve_indices = torch.ones((len(leaf_indices), max_depth), dtype=torch.long, device=device) * -1
    
    # Fill in the actual indices
    for i, path in enumerate(paths):
        retrieve_indices[i, :len(path)] = torch.tensor(path)
    
    return retrieve_indices

def construct_tree_mask(parents, device):
    tree_mask = torch.zeros((1, 1, len(parents), len(parents)), dtype=torch.float32, device=device)
    for i, parent in enumerate(parents):
        if parent != -1:
            tree_mask[..., i, :] = tree_mask[..., parent, :]
        else:
            assert i == 0
        tree_mask[..., i, i] = 1
    return tree_mask

def construct_tree_position_ids(parents, device):
    tree_position_ids = torch.zeros(len(parents), dtype=torch.long, device=device)
    for i, parent in enumerate(parents):
        if parent == -1:
            tree_position_ids[i] = 0
        else:
            tree_position_ids[i] = tree_position_ids[parent] + 1
    return tree_position_ids

def hybrid_speculate(model, input_ids, hidden_states, logits_processor):
    """Speculates the next tokens using either the suffix tree or EAGLE-3.

    Args:
        model (HybridModel): The model containing the LLM and the EAGLE head
        input_ids (torch.Tensor): The input token IDs, including the bonus token
        hidden_states (torch.Tensor): The concatenated 2nd/mid/2nd-to-last hidden states from the LLM, excluding the bonus token
        logits_processor (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert hasattr(model, "_suffix_cache"), "SuffixCache not initialized. Please call init_suffix_cache() first."
    result = model._suffix_cache.speculate(
        0,
        input_ids[0].tolist(),
        max_spec_tokens=model.draft_num_candidate_tokens,
        max_spec_factor=model.max_spec_factor,
        min_token_prob=model.min_token_prob
    )
    speculate_with_suffix = result.score >= model.use_suffix_threshold

    # print("speculated_with_suffix", speculate_with_suffix)
    assert hasattr(model, "suffix_hidden_states"), "suffix_hidden_states not initialized"
    if model.suffix_hidden_states is not None:
        # print(f"appending model.suffix_hidden_states ({model.suffix_hidden_states.shape}) to hidden_states ({hidden_states.shape})")
        hidden_states = torch.cat((model.suffix_hidden_states, hidden_states), dim=-2)
    if speculate_with_suffix:
        model.suffix_hidden_states = hidden_states
        # print("setting model.suffix_hidden_states to", model.suffix_hidden_states.shape)
    else:
        model.suffix_hidden_states = None
        # print("clearing model.suffix_hidden_states to " , model.suffix_hidden_states)
    assert hidden_states is not None, "hidden_states is None"


    if speculate_with_suffix:
        draft_tokens = torch.tensor(result.token_ids, dtype=torch.int64, device=input_ids.device).unsqueeze(0)
        retrieve_indices = construct_retrieve_indices(result.token_ids, result.parents, input_ids.device)
        tree_mask = construct_tree_mask(result.parents, input_ids.device)
        tree_position_ids = construct_tree_position_ids(result.parents, input_ids.device)
        # print("\n-------- suffix tree speculation --------")
        # print("input_ids", input_ids.shape, input_ids.dtype)
        # print("\thidden_states",hidden_states.shape, hidden_states.dtype)
        # print("draft_tokens", draft_tokens.shape, draft_tokens.dtype)
        # print("\t",draft_tokens)
        # print("retrieve_indices", retrieve_indices.shape, retrieve_indices.dtype)
        # print("\t",retrieve_indices)
        # print("tree_mask", tree_mask.shape, tree_mask.dtype)
        # print("\t",tree_mask)
        # print("tree_position_ids", tree_position_ids.shape, tree_position_ids.dtype)
        # print("\t",tree_position_ids)
        # print()
    else:
        # If the score is below the threshold, use EAGLE-3 to speculate
        # print("\n-------- eagle3 speculation -------- ")
        # print("\tinput_ids", input_ids.shape, input_ids.dtype)
        # print("\thidden_states",hidden_states.shape, hidden_states.dtype)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(hidden_states, input_ids, model.base_model.lm_head,logits_processor)
        # print("draft_tokens", draft_tokens.shape, draft_tokens.dtype)
        # print("\t",draft_tokens)
        # print("retrieve_indices", retrieve_indices.shape, retrieve_indices.dtype)
        # print("\t",retrieve_indices)
        # print("tree_mask", tree_mask.shape, tree_mask.dtype)
        # print("\t",tree_mask)
        # print("tree_position_ids", tree_position_ids.shape, tree_position_ids.dtype)
        # print("\t",tree_position_ids)
        # print()
        
    
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
    

def initialize_tree(input_ids, model, past_key_values, logits_processor):
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )

    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    # Clone the output hidden states
    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states=torch.cat(outputs["hidden_states"],dim=-1)
    
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = hybrid_speculate(model, input_ids, hidden_states, logits_processor)
    
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig


def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]

    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state





def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    
    # update suffix tree
    model._suffix_cache.update_response(0, candidates[best_candidate, : accept_length + 1].tolist())

    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])
    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    # print("@ update_inference_inputs: @")
    # print(f"input_ids - adding accepted tokens ({accept_length.item()}): {prev_input_len} -> {input_ids.shape[1]}")
    # print("hidden_state_new", hidden_state_new.shape, hidden_state_new.dtype)
    # print("retrieve_indices", retrieve_indices.shape, retrieve_indices.dtype)
    # print(retrieve_indices)
    # print("retrieve_hidden_state_new", retrieve_hidden_state_new.shape, retrieve_hidden_state_new.dtype)
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    # print("best_candidate", best_candidate)
    # print("accept_length", accept_length)
    # print("accept_hidden_state_new", accept_hidden_state_new.shape, accept_hidden_state_new.dtype)
    
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    # print("accept_length", accept_length.item())
    input_ids_new = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    # print("input_ids_new: ", input_ids_new.shape)

    
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = hybrid_speculate(model, 
                                                                                    input_ids=input_ids_new,
                                                                                    hidden_states=accept_hidden_state_new,
                                                                                    logits_processor=logits_processor)

    new_token += accept_length + 1

    return input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
