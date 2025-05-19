import copy
from dataclasses import dataclass, field
from typing import Hashable, List, Optional, Tuple, Sequence, Union

import torch

from .suffix_decoding import SuffixTree, Candidate

from transformers.utils import ModelOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import _crop_past_key_values
import json
from transformers import AutoTokenizer
from tqdm import tqdm

@dataclass
class SuffixSpecResult:
    """
    A dataclass representing the result of a speculation using SuffixDecoding.

    Attributes:
        token_ids (List[int]): List of token IDs in the speculation result.
        parents (List[int]): List of parent indices for each token used to
            encode the tree structure. The parent token of token_ids[i] is
            token_ids[parents[i]].
        probs (List[float]): List of estimated probabilities for each token.
        score (float): The overall score of the suffix match computed as the
            sum of the estimated probabilities of each speculated token.
        match_len (int): The length of the pattern match that yielded this
            speculation result.
    """
    token_ids: List[int] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)
    probs: List[float] = field(default_factory=list)
    score: float = 0.0
    match_len: int = 0

    @staticmethod
    def from_candidate(candidate: Candidate) -> "SuffixSpecResult":
        return SuffixSpecResult(
            token_ids=candidate.token_ids,
            parents=candidate.parents,
            probs=candidate.probs,
            score=candidate.score,
            match_len=candidate.match_len,
        )


class SuffixCache:
    
    def __init__(self, max_depth: int = 64, training_file: Optional[str] = None, tokenizer: Optional[AutoTokenizer] = None):
        self._max_depth = max_depth
        self._suffix_tree = SuffixTree(max_depth)
        self._prompt_trees = {}
        self._req_to_seq_id = {}
        if training_file is not None:
            assert tokenizer is not None, "Tokenizer must be provided if training file is provided"
            self.load_training_file(training_file, tokenizer)

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def has_cached_prompt(self, req_id: Hashable) -> bool:
        return req_id in self._prompt_trees

    def cached_prompt_ids(self) -> List[Hashable]:
        return list(self._prompt_trees.keys())
    
    def load_training_file(self, training_file: str, tokenizer: AutoTokenizer):
        with open(training_file, "r") as f:
            total_lines = sum(1 for _ in f)
            f.seek(0)
            for i, line in enumerate(tqdm(f, desc="Loading training file", total=total_lines)):
                entry = json.loads(line)
                prompt = entry["prompt"]
                response = entry["response"]
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                response_tokens = tokenizer.encode(response, add_special_tokens=False)
                self.cache_prompt(1000000 + i, prompt_tokens)
                self.update_response(1000000 + i, response_tokens)

    def cache_prompt(self, req_id: Hashable, prompt_token_ids: Sequence[int]):
        """
        Cache a prompt for a specific request ID. Future speculations for the
        same request may also source draft tokens from this prompt.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.
            prompt_token_ids (Sequence[int]): A sequence of token IDs
                representing the prompt to be cached.

        Raises:
            ValueError: If a prompt already exists for the given request ID.

        Note:
            The caller should evict the cached prompt using `evict_prompt` once
            the prompt is no longer needed (i.e. the request is completed).
        """
        if req_id in self._prompt_trees:
            raise ValueError(f"Prompt already exists for request '{req_id}'")
        self._prompt_trees[req_id] = SuffixTree(self._max_depth)
        self._prompt_trees[req_id].extend(0, prompt_token_ids)

    def evict_prompt(self, req_id: Hashable):
        """
        Evicts a prompt from the cache for a specific request.

        Args:
            req_id (Hashable): The unique identifier for the request whose
                prompt should be evicted.

        Raises:
            ValueError: If no prompt exists for the given request identifier.
        """
        if req_id not in self._prompt_trees:
            raise ValueError(f"Prompt does not exist for request '{req_id}'")
        del self._prompt_trees[req_id]

    def _get_or_assign_seq_id(self, req_id: Hashable) -> int:
        if req_id not in self._req_to_seq_id:
            self._req_to_seq_id[req_id] = len(self._req_to_seq_id)
        return self._req_to_seq_id[req_id]

    def update_response(
        self,
        req_id: Hashable,
        token_ids: Union[int | Sequence[int]],
    ):
        """
        Update the cached response for a given request by adding token(s) to
        its end. It does not rely on the prompt being cached for the request,
        and its lifetime does not depend on the prompt's existence. Once the
        response is updated, the new tokens can be used for future speculations
        for all requests.

        Args:
            req_id (Hashable): The unique identifier for the request.
            token_ids (Union[int, Sequence[int]]): Either a single token ID
                (int) or a sequence of token IDs to be appended to the response
                for the given request.

        Notes:
            - If req_id doesn't exist, a new empty sequence will be initialized.
            - If token_ids is a single integer, it's added as a single token.
            - If token_ids is a sequence, all tokens in the sequence are added.
        """
        seq_id = self._get_or_assign_seq_id(req_id)
        if isinstance(token_ids, int):
            self._suffix_tree.append(seq_id, token_ids)
            if req_id in self._prompt_trees:
                self._prompt_trees[req_id].append(0, token_ids)
        else:
            self._suffix_tree.extend(seq_id, token_ids)
            if req_id in self._prompt_trees:
                self._prompt_trees[req_id].extend(0, token_ids)

    def speculate(
        self,
        req_id: Hashable,
        pattern: Sequence[int],
        max_spec_tokens: int = 0,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
        use_cached_prompt: bool = True,
    ) -> SuffixSpecResult:
        """
        Speculates and returns the most likely continuation of a given token
        pattern using the request-specific prompt cache (if available) and the
        global cache of previous responses.

        Args:
            req_id (Hashable): The unique identifier for the request.
            pattern (Sequence[int]): The sequence of token IDs to match and
                continue from.
            max_spec_tokens (int): Maximum number of tokens to speculate. If 0,
                uses the cache's max_depth.
            max_spec_factor (float): Factor that limits speculation based on
                matched pattern length.
            min_token_prob (float): Minimum estimated probability threshold for
                candidate tokens.
            use_tree_spec (bool): If True, uses tree-based speculation.
            use_cached_prompt (bool): If True, uses the cached prompt for the
                request in addition to the global cache of previous responses.
        
        Returns:
            The speculation result containing the most likely continuation
            tokens, their probabilities, and overall score.

        Raises:
            ValueError: If the prompt doesn't exist for the given req_id when
                use_cached_prompt is True, or if the pattern is invalid.
        """
        if use_cached_prompt and req_id not in self._prompt_trees:
            raise ValueError(f"Prompt does not exist for request '{req_id}'")
        if not pattern:
            raise ValueError("Pattern must not be empty")

        if not max_spec_tokens:
            max_spec_tokens = self.max_depth

        if len(pattern) > self._max_depth:
            pattern = pattern[-self._max_depth :]

        if use_cached_prompt:
            prompt_tree = self._prompt_trees[req_id]
            candidate = prompt_tree.speculate(
                pattern,
                max_spec_tokens,
                max_spec_factor,
                min_token_prob,
                use_tree_spec)
            result = SuffixSpecResult.from_candidate(candidate)
        else:
            result = SuffixSpecResult()

        candidate = self._suffix_tree.speculate(
            pattern,
            max_spec_tokens,
            max_spec_factor,
            min_token_prob,
            use_tree_spec)
        if candidate.score > result.score:
            result = SuffixSpecResult.from_candidate(candidate)
        return result



@torch.no_grad()
def greedy_search_suffix(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        draft_matching_window_size=3,
        draft_num_candidate_tokens=10,
        **model_kwargs,
):
    assert hasattr(self, "_suffix_cache"), "SuffixCache not initialized. Please call init_suffix_cache() first."
    # if not hasattr(self, "_suffix_cache"):
    #     self._suffix_cache = SuffixCache(64)
    global tokenizer

    # init values
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None

    max_len = stopping_criteria[0].max_length

    self._suffix_cache.cache_prompt(0, input_ids[0].tolist())

    step = 0
    accept_length_list = []
    while True:
        step += 1
        # input_ids: committed tokens (or prompt)
        cur_len = input_ids.shape[-1]

        # lookup prefix and predict
        # candidate_pred_tokens = find_candidate_pred_tokens(input_ids, draft_matching_window_size,
        #                                                    draft_num_candidate_tokens)
        result = self._suffix_cache.speculate(
            0,
            input_ids[0].tolist(),
            max_spec_tokens=draft_num_candidate_tokens,
            max_spec_factor=2.0,
            min_token_prob=0.1
        )
        candidate_pred_tokens = torch.tensor(result.token_ids, device=input_ids.device).unsqueeze(0)

        if len(result.token_ids) > 0:
            candidate_input_ids = torch.cat((input_ids, candidate_pred_tokens), dim=1)
        else:
            candidate_input_ids = input_ids

        # candidate_input_ids = torch.cat((input_ids, candidate_pred_tokens), dim=1)
        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        candidate_kwargs = copy.copy(model_kwargs)

        attention_mask = candidate_kwargs["attention_mask"]
        if attention_mask is not None:
            mask_extension_length = candidate_input_ids.shape[1] - attention_mask.shape[1]
            candidate_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], mask_extension_length))], dim=-1,)
        # if candidate_length > 0:
        #     print("attention_mask.shape", attention_mask.shape)
        #     print("Adding: attention_mask.new_ones((attention_mask.shape[0], candidate_length))", attention_mask.new_ones((attention_mask.shape[0], candidate_length)).shape)
        #     candidate_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], candidate_length))], dim=-1,)
        #     print("resulting attention_mask.shape", candidate_kwargs["attention_mask"].shape)
        # else:
        #     candidate_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1,)
        candidate_kwargs = self._get_initial_cache_position(candidate_input_ids, candidate_kwargs)
        # print("candidate_input_ids", candidate_input_ids)
        # print("candidate_kwargs", candidate_kwargs)
        model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)

        # print("step: ", step)
        # print("candidate_length: ", candidate_length)
        # print("input_ids.shape: ", input_ids.shape)
        # print()

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # print()
        # print("candidate_input_ids", candidate_input_ids)
        # print("candidate_input_ids.shape", candidate_input_ids.shape)

        # print("model_inputs", model_inputs)
        # # print("model_inputs.shape", model_inputs.shape)
        # print("outputs.logits.shape", outputs.logits.shape)
        # print("candidate_length", candidate_length)

        new_logits = outputs.logits[:, -candidate_length - 1:]  # excludes the input prompt if present
        selected_tokens = new_logits.argmax(dim=-1)
        candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
        # print("selected_tokens", selected_tokens)
        # print("selected_tokens.shape", selected_tokens.shape)
        # print("candidate_new_tokens", candidate_new_tokens)
        # print("candidate_new_tokens.shape", candidate_new_tokens.shape)
        # print("--------------------------------")
        if candidate_length > 0:
            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
        else:
            n_matches = 0

        n_matches = min(n_matches, max_len - cur_len - 1)
        # print(f"step: {step}, candidate_length: {candidate_length}, accepted: {n_matches.item() if isinstance(n_matches, torch.Tensor) else n_matches}")

        valid_tokens = selected_tokens[:, : n_matches + 1]
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        new_cur_len = input_ids.shape[-1]

        self._suffix_cache.update_response(0, valid_tokens[0].tolist())

        new_cache_size = new_cur_len - 1
        # print("new_cache_size: ", new_cache_size)
        outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

        model_kwargs["past_key_values"] = outputs.past_key_values

        accept_length_tree = new_cur_len - cur_len
        accept_length_list.append(accept_length_tree)

        # stop if we exceed the maximum length

        if (valid_tokens == eos_token_id_tensor.item()).any():
            break

        if stopping_criteria(input_ids, scores):
            break
    
    self._suffix_cache.evict_prompt(0)
    
    idx = step - 1
    return input_ids, idx, accept_length_list