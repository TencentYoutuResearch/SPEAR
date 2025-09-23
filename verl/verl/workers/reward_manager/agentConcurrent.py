# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.workers.reward_manager import register


def has_repeated_ngrams(words, n=20, threshold=10):
    # words = text.split()
    words = [str(item) for item in words]
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = Counter(ngrams)
    return any(count >= threshold for count in counts.values())



def get_file_path(download_dir):
    if type(download_dir) is not str:
        print("Download_dir:", download_dir)
        try:
            download_dir = str(download_dir)
        except:
            print(f"Failed to convert download_dir from {type(download_dir)} to str")
            return None

    try:
        not_downloaded_files = ["prompt.jsonl"]
        file_names = os.listdir(download_dir)
        # assert len(file_names) <= len(not_downloaded_files) + 1,
        # f"Expected at most 1 downloaded file, got {len(file_names)}"
        if len(file_names) == 0:
            return None
        file_path = None
        for name in file_names:
            if name not in not_downloaded_files:
                file_path = os.path.join(download_dir, name)
                break

        return file_path
    except Exception as e:
        # Print error details including variable types and values
        print(f"Error processing directory: {download_dir} (type: {type(download_dir)})")
        print(f"Exception type: {type(e).__name__}, Message: {str(e)}")
        # print(f"len(absolute_paths) = {len(absolute_paths)}")
        return None


@register("agentConcurrent")
class AgentConcurrentRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None,\
        reward_fn_key="data_source", **reward_kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def process_data_item(self, data_item, i):
        """
        data_item: DataProtoItem
        """
        prompt_ids = data_item.batch["prompts"]
        if "messages" in data_item.non_tensor_batch["messages"]:
            messages = data_item.non_tensor_batch["messages"]['messages']
        else:
            messages = []
        if "tools_available" in data_item.non_tensor_batch["messages"]:
            tools = data_item.non_tensor_batch["messages"]['tools_available']
        elif "tools" in data_item.non_tensor_batch["messages"]:
            tools = data_item.non_tensor_batch["messages"]['tools']
        else:
            tools = None

        global_steps = float(data_item.meta_info["global_steps"])
        use_toolcall_reward = data_item.meta_info["use_toolcall_reward"]
        max_toolcall_steps = data_item.meta_info["max_toolcall_steps"]

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        ground_truth = data_item.non_tensor_batch["reward_model"].get("ground_truth", None)

        data_source = data_item.non_tensor_batch[self.reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", None)

        if extra_info is not None and type(extra_info) is dict and "question" in extra_info:
            user_query = data_item.non_tensor_batch["extra_info"]["question"]
        else:
            user_query = ""

        if "download_list" in data_item.non_tensor_batch:
            download_dir = data_item.non_tensor_batch["download_list"]
            if data_source in ["web_download"]:
                file_path = get_file_path(download_dir)
            else:
                file_path = None
        else:
            download_dir = None
            file_path = None

        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["num_turns"] = num_turns

        score = self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            user_query=user_query,
            file_path=file_path,
            extra_info=extra_info,
            tools=tools,
            messages=messages,
            global_steps=global_steps,
            use_toolcall_reward=use_toolcall_reward,
            max_toolcall_steps=max_toolcall_steps,
            **self.reward_kwargs,
        )
        if data_source in ["web_donwload"]:
            print(f"🌏 Download score: {score}, 🗂️ Download_dir: {download_dir}")
        return i, score, valid_response_length, valid_response_ids, data_source,\
            prompt_str, response_str, ground_truth, extra_info, user_query, download_dir, file_path


    def __call__(self, data: DataProto, return_dict=False, max_response_len=16384):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        reward_extra_info["is_incomplete"] = []
        reward_extra_info["is_overlong"] = []

        already_print_data_sources = {}
        rewards = []
        results_all_by_idx = {}
        print("Start Agent Reward 🥇 Forwarding")
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = {executor.submit(self.process_data_item, data[i], i): i for i in range(len(data))}
            for future in tqdm(as_completed(futures), total=len(futures)):
                i, score, valid_response_length, valid_response_ids, data_source, prompt_str,\
                    response_str,ground_truth, extra_info, user_query, download_dir, file_path = future.result()
                results_all_by_idx[i] = (score, valid_response_length, valid_response_ids,\
                    data_source, prompt_str, response_str,\
                        ground_truth, extra_info, user_query, download_dir, file_path)

        for i in range(len(data)):
            score, valid_response_length, valid_response_ids, data_source, prompt_str, response_str,\
                ground_truth, extra_info, user_query, download_dir, file_path = results_all_by_idx[i]
            is_incomplete = 0
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

            if isinstance(score, dict):
                score["score"] = np.float64(score["score"])
                reward = score["score"]
                if "is_incomplete" not in score:
                    score["is_incomplete"] = is_incomplete
                if "is_overlong" not in score:
                    score["is_overlong"] = is_overlong
                if "is_repetitive" not in score:
                    score["is_repetitive"] = is_repetitive
                if "is_unreadable" not in score:
                    score["is_unreadable"] = is_unreadable
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                score = np.float64(score)
                reward = score
                reward_extra_info["is_incomplete"].append(is_incomplete)
                reward_extra_info["is_overlong"].append(is_overlong)
                reward_extra_info["is_repetitive"].append(is_repetitive)
                reward_extra_info["is_unreadable"].append(is_unreadable)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("📝 [prompt]", prompt_str)
                print("📨 [response]", response_str)
                print("✅ [ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("📈 [score]", score)

        if "acc" in reward_extra_info:
            reward_extra_info["acc"] = [float(acc_item) for acc_item in reward_extra_info["acc"]]
        # import pdb; pdb.set_trace()
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
