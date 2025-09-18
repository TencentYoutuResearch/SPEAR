# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
import torch
import numpy as np
from collections import defaultdict
import re
from collections import Counter
import os
from tqdm import tqdm



def has_repeated_ngrams(words, n=20, threshold=10):
    # words = text.split()
    words = [str(item) for item in words]
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = Counter(ngrams)
    return any(count >= threshold for count in counts.values())



def is_think_action_valid(solution_str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Judge if the content is none
    if solution_str is None:
        return False
    if len(str(solution_str).strip()) == 0:
        return False

    # Extract think content using XML-style tags
    think_pattern = r'<think>(.*?)</think>'
    matches_think = list(re.finditer(think_pattern, solution_str, re.DOTALL))

    if matches_think:
        final_think = matches_think[-1].group(1).strip()
    else:
        final_think = None

    if "</think>" in solution_str:
        answer_candidate = (solution_str[solution_str.rfind("</think>")+len("</think>"):]).strip()
    else:
        answer_candidate = solution_str
    
    # Extract think content using XML-style tags
    action_pattern = r'<action>(.*?)</action>'
    matches_action = list(re.finditer(action_pattern, answer_candidate, re.DOTALL))
    if matches_action:
        final_answer = matches_action[-1].group(1).strip()
    else:
        final_answer = None

    # if (final_think is None) or (final_answer is None):
    #     return False
    # else:
    #     return True

    if final_answer is None:
        return False
    else:
        return True




class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, normalize_by_length=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length

    def __call__(self, data: DataProto, return_dict=False, max_response_len=16384):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        reward_extra_info["is_incomplete"] = []
        reward_extra_info["is_overlong"] = []
        reward_extra_info["is_repetitive"] = []
        reward_extra_info["is_unreadable"] = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            # ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            multi_modal_inputs = data_item.non_tensor_batch.get('multi_modal_inputs', None)
            if multi_modal_inputs is not None:
                pixel_values = multi_modal_inputs['pixel_values']
                image_grid_thw = multi_modal_inputs['image_grid_thw']


            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']

            if self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards
            ## 是否施加step-wise reward
            use_toolcall_reward = data_item.meta_info['use_toolcall_reward']
            max_toolcall_steps = data_item.meta_info['max_toolcall_steps']
            if use_toolcall_reward == "none":
                score += 0  # 不额外添加toolcall-reward
            elif use_toolcall_reward == "constant":
                # 最多10轮
                score_toolcall = min(1.0, episode_lengths * 0.1)
                score += score_toolcall * 1
            else:
                # cosine衰减
                global_steps = data_item.meta_info['global_steps']
                assert(global_steps >= 0)
                if global_steps <= max_toolcall_steps:
                    score_toolcall = min(1.0, episode_lengths * 0.1)
                    score_toolcall *= (np.cos((global_steps / max_toolcall_steps) * np.pi) + 1)/2
                    score_toolcall = max(0, score_toolcall)
                    score += score_toolcall

            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            is_overlong = 0
            if valid_response_length >= max_response_len:
                is_overlong = 1
            is_repetitive = 0
            if has_repeated_ngrams(valid_response_ids, n=20, threshold=10):
                is_repetitive = 1
            is_unreadable = 0
            for invalid_tk_str in ["�", "ï¿½", "�", "ï¼�", "ï¼Ÿ", "ï¼Ÿ", "ï¿½ï¿½", "ï¿½ï¿½ï¿½ï¿½", "Ġï¿½"]:
                if invalid_tk_str in response_str:
                    is_unreadable = 1
                    break
            is_incomplete = 0
            if not is_think_action_valid(response_str):
                is_incomplete = 1

            reward_extra_info["is_incomplete"].append(is_incomplete)
            reward_extra_info["is_overlong"].append(is_overlong)
            reward_extra_info["is_repetitive"].append(is_repetitive)
            reward_extra_info["is_unreadable"].append(is_unreadable)
            # import pdb;pdb.set_trace();
            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
