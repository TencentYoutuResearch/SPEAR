# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
Note that this hybrid mode means the online rollout trajectories are added into SFT training
"""
from copy import deepcopy
import json
import os
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional
from collections import defaultdict
import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from PIL import Image
from verl import DataProto
# from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type
from tensordict import TensorDict
import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.multiturn_sft_dataset_online import MultiTurnSFTDatasetOnline

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import DataLoader, Dataset, DistributedSampler
WorkerType = Type[Worker]
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask, _timer, apply_invalid_action_penalty
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from torch import nn, optim
from copy import deepcopy
import random
import numpy as np

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch


class TrajectoryBuffer:
    """
    存储所有轨迹 并动态删除老轨迹
    """
    def __init__(self, tokenizer, buffer_size=2048, tolerate_steps=10, threshold=1.0):
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.tolerate_steps = min(tolerate_steps, 10)
        self.trajectory_buffer = []
        self.threshold = threshold
        self.last_train_step = 0

    def update(self, tools, messages, global_step):
        if len(self.trajectory_buffer) >= self.buffer_size:
            self.trajectory_buffer.pop(0)
        self.trajectory_buffer.append([tools, messages, global_step])

    def remove_old(self):
        if len(self.trajectory_buffer):
            step_latest = self.trajectory_buffer[-1][-1]
            self.trajectory_buffer = [item for item in self.trajectory_buffer if abs(item[-1]-step_latest)<=self.tolerate_steps]

    def is_full_capacity(self):
        if self.get_buffer_size() >= self.buffer_size:
            return True
        else:
            return False

    def get_buffer(self):
        tools = [item[0] for item in self.trajectory_buffer]
        messages = [item[1] for item in self.trajectory_buffer]
        return tools, messages

    def reset_buffer(self):
        self.trajectory_buffer = []
        
    def update_batch(self, batch_messages, batch_scores, batch_response_masks, global_step):
        cur_batch_valid = []
        count_by_prompt = {}
        for batch_message, batch_score, batch_response_mask in zip(batch_messages, batch_scores, batch_response_masks):
            # 使用advantage来判断是否大于阈值
            if batch_score < self.threshold:
                continue
            if batch_response_mask == 0:
                continue
            # 必须要包含工具调用
            tools = batch_message.get("tools", None)
            messages = batch_message["messages"]
            prompt = messages[0]["content"]
            if not (prompt in count_by_prompt):
                count_by_prompt[prompt] = 0
            count_by_prompt[prompt] += 1
            has_tool_call = False
            try:
                for message in messages:
                    if "tool_calls" in message and message["tool_calls"] is not None:
                        has_tool_call = True
                    assert("role" in message)
                    assert("content" in message)
            except Exception as e:
                print("Found errors in messages", e)
                print(f"❌ dame Messages:\n\n{messages}")
                continue
            if not has_tool_call:
                continue
            cur_batch_valid.append([tools, messages])
        # introduce randomness
        random.shuffle(cur_batch_valid)
        for valid_tools_messages in cur_batch_valid:
            tools, messages = valid_tools_messages
            prompt = messages[0]["content"]
            # here we only use the prompt for counting and 
            # keep those with at least two different responses
            if prompt in count_by_prompt and count_by_prompt[prompt] > 1:
                self.update(tools, messages, global_step)

    def get_buffer_size(self):
        return len(self.trajectory_buffer)



class TrajectoryBufferBatch:
    """
    存储所有轨迹包括计算好的所有内容; 按照batch处理; 并动态删除老轨迹
    """
    def __init__(self, tokenizer, buffer_size=2048, baseline_buffer_size=10240, tolerate_steps=10, weight_decay_trajectory_replay=0.9,\
        adv_estimator=AdvantageEstimator.GRPO, norm_adv_by_std_in_grpo=False):
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.baseline_buffer_size = baseline_buffer_size
        self.tolerate_steps = min(tolerate_steps, 10)
        self.trajectory_buffer = []
        self.reward_mean_history = []
        self.reward_std_history = []
        self.last_train_step = 0
        self.weight_decay_trajectory_replay = weight_decay_trajectory_replay
        self.adv_estimator = adv_estimator
        self.norm_adv_by_std_in_grpo = norm_adv_by_std_in_grpo

    def remove_old(self):
        if len(self.trajectory_buffer):
            step_latest = self.trajectory_buffer[-1][-1]
            self.trajectory_buffer = [item for item in self.trajectory_buffer if abs(item[-1]-step_latest)<=self.tolerate_steps]
        return

    def is_full_capacity(self):
        if self.get_buffer_size() > self.buffer_size * 1.5:
            ## only times 1.5 for buffer
            return True
        else:
            return False

    def maintain_reward_statistics(self, step_maximum, step_list):
        ## 维护所有历史step压平后的50分位数reward & std
        reward_mean_history_flatten = []
        reward_std_history_flatten = []
        ## 维护每个step的50分位数reward & std
        reward_mean_history_step = {}
        reward_std_history_step = {}

        for reward_mean_history_item in self.reward_mean_history:
            reward_mean_history_flatten += reward_mean_history_item[1]
            reward_mean_history_step[reward_mean_history_item[0]] = np.percentile(reward_mean_history_item[1], 50)

        for reward_std_history_item in self.reward_std_history:
            reward_std_history_flatten += reward_std_history_item[1]
            reward_std_history_step[reward_std_history_item[0]] = np.percentile(reward_std_history_item[1], 50)
        
        reward_mean_maximum_step = reward_mean_history_step[step_maximum]
        reward_std_maximum_step = reward_std_history_step[step_maximum]
        reward_mean_flatten = [v for k, v in reward_mean_history_step.items()]
        reward_std_flatten = [v for k, v in reward_std_history_step.items()]
        reward_mean_maximum = max(reward_mean_flatten)
        reward_std_maximum = max(reward_std_flatten)
        reward_mean_95p = np.percentile(reward_mean_history_flatten, 95)
        reward_std_95p = np.percentile(reward_std_history_flatten, 95)
        reward_mean_50p = np.percentile(reward_mean_history_flatten, 50)
        reward_std_50p = np.percentile(reward_std_history_flatten, 50)
        custom_reward_mean_list = []
        custom_reward_std_list = []
        for reward_step in step_list:
            # 维护列表选取50 percentile的上届
            # custom_reward_mean_list.append(max(reward_mean_maximum_step, reward_mean_history_step[reward_step]))
            # custom_reward_std_list.append(max(reward_std_maximum_step, reward_std_history_step[reward_step]))
            custom_reward_mean_list.append(max(reward_mean_maximum, reward_mean_history_step[reward_step]))
            custom_reward_std_list.append(max(reward_std_maximum, reward_std_history_step[reward_step]))
        print(f"""🐹 计算step-wise reward batch mean percentile {reward_mean_history_step}
        with current:{reward_mean_maximum_step}/p50:{reward_mean_50p}/maximum:{reward_mean_maximum}""")
        print(f"""🐹 计算step-wise reward batch std percentile {reward_std_history_step}
        with current:{reward_std_maximum_step}/p50:{reward_std_50p}/maximum:{reward_std_maximum}""")
    
        return reward_mean_history_flatten, reward_std_history_flatten,\
            reward_mean_history_step, reward_std_history_step, reward_mean_maximum_step,\
                reward_std_maximum_step, reward_mean_maximum, reward_std_maximum,\
                    reward_mean_95p, reward_std_95p, reward_mean_50p, reward_std_50p,\
                        custom_reward_mean_list, custom_reward_std_list


    def prepare_for_selection(self, batch_concat, reward_tensor_concat, reward_extra_info_dict_concat,\
        reward_mean_50p, reward_std_50p):
        if self.weight_decay_trajectory_replay <= 0 or self.weight_decay_trajectory_replay > 1:
            # here we use the recomputed reward for new advantages
            assert(len(custom_reward_mean_list) == len(step_list))
            assert(len(custom_reward_std_list) == len(step_list))
            # 重新计算后仅保留正样本
            batch_concat.batch["token_level_scores"] = reward_tensor_concat
            batch_concat.batch["token_level_rewards"] = reward_tensor_concat
            # must be grpo
            # assert self.adv_estimator == AdvantageEstimator.GRPO
            # Initialize the mask for GRPO calculation
            # Call compute_grpo_outcome_advantage with parameters matching its definition
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=batch_concat.batch["token_level_rewards"],
                response_mask=batch_concat.batch["response_mask"],
                index=batch_concat.non_tensor_batch["uid"],
                traj_index=batch_concat.non_tensor_batch['traj_uid'],
                norm_adv_by_std_in_grpo=self.norm_adv_by_std_in_grpo,
                custom_reward_mean_std={"mean": reward_mean_50p, "std": reward_std_50p}
                # custom_reward_mean_std={"mean": custom_reward_mean_list, "std": custom_reward_std_list}
            )
            batch_concat.batch["advantages"] = advantages
            batch_concat.batch["returns"] = returns
            advantages_len = advantages.size(0)
            mask = (batch_concat.batch["advantages"].mean(-1) >= 0)
            batch_concat.batch = batch_concat.batch[mask]
            reward_tensor_concat = reward_tensor_concat[mask]
            advantages_len_filter = batch_concat.batch.size(0)
            print(f"👹 Now recalculating the advantage size={advantages_len} and filter out those negative ones size={advantages_len_filter}")
            for key, value in batch_concat.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    batch_concat.non_tensor_batch[key] = value[mask]
                elif isinstance(value, list) or isinstance(value, tuple):
                    batch_concat.non_tensor_batch[key] = [v for v, m in zip(value, mask) if m]
                else:
                    batch_concat.non_tensor_batch[key] = value

            for key, value in batch_concat.meta_info.items():
                if isinstance(value, np.ndarray):
                    batch_concat.meta_info[key] = value[mask]
                elif isinstance(value, list) or isinstance(value, tuple):
                    batch_concat.meta_info[key] = [v for v, m in zip(value, mask) if m]
                else:
                    batch_concat.meta_info[key] = value

            for key, value in reward_extra_info_dict_concat.items():
                if isinstance(value, np.ndarray):
                    reward_extra_info_dict_concat[key] = value[mask]
                elif isinstance(value, list) or isinstance(value, tuple):
                    reward_extra_info_dict_concat[key] = [v for v, m in zip(value, mask) if m]
                else:
                    reward_extra_info_dict_concat[key] = value

        elif self.weight_decay_trajectory_replay > 0 and self.weight_decay_trajectory_replay <= 1:
            # here we apply advantage weight decay to the original advantages
            batch_concat.batch["advantages"] *= self.weight_decay_trajectory_replay
        return batch_concat
        

    def get_buffer(self):
        def nearest_lower_power_of_2(n):
            # 只抽取小于等于该数的幂级数
            if n <= 1:
                return 0
            return 1 << (n.bit_length() - 1)

        def largest_ratio_32(n):
            if n <= 32:
                return 0
            return 32 * (n // 32)

        assert (len(self.trajectory_buffer) >= 1)
        # [batch, reward_tensor, reward_extra_infos_dict, global_step]
        step_list = []
        for traj_pair in self.trajectory_buffer:
            step_global = traj_pair[-1]
            step_list += [step_global] * len(traj_pair[1])
        step_maximum = max(step_list)
        # 整个step的最大值就是最新一轮的step的batch
        batch_list = [traj_pair[0] for traj_pair in self.trajectory_buffer]
        # import pdb;pdb.set_trace();
        batch_concat = DataProto.concat(batch_list)
        reward_tensor_list = [traj_pair[1] for traj_pair in self.trajectory_buffer]
        reward_tensor_concat = torch.cat(reward_tensor_list, dim=0)
        reward_extra_infos_dict_list = [traj_pair[2] for traj_pair in self.trajectory_buffer]
        reward_extra_info_dict_concat = defaultdict(list)
        print("Before Adv calculation in TrajectoryBuffer", "len(step_list)", len(step_list), "len(reward_tensor_concat)", len(reward_tensor_concat))
        assert(len(step_list) == len(reward_tensor_concat))
        # import pdb;pdb.set_trace();
        for reward_extra_infos_dict in reward_extra_infos_dict_list:
            for key, value in reward_extra_infos_dict.items():
                if not (key in reward_extra_info_dict_concat):
                    reward_extra_info_dict_concat[key] = value
                    continue
                if isinstance(value, np.ndarray):
                    reward_extra_info_dict_concat[key] = np.concatenate([reward_extra_info_dict_concat[key], value], axis=0)
                elif isinstance(value, list):
                    reward_extra_info_dict_concat[key] = reward_extra_info_dict_concat[key] + value
                elif isinstance(value, tuple):
                    reward_extra_info_dict_concat[key] = tuple(list(reward_extra_info_dict_concat[key]) + list(value))
                else:
                    reward_extra_info_dict_concat[key] = value

        # compute the statistics
        reward_mean_history_flatten, reward_std_history_flatten,\
                    reward_mean_history_step, reward_std_history_step, reward_mean_maximum_step,\
                        reward_std_maximum_step, reward_mean_maximum, reward_std_maximum,\
                            reward_mean_95p, reward_std_95p, reward_mean_50p, reward_std_50p,\
                                custom_reward_mean_list, custom_reward_std_list = \
                                    self.maintain_reward_statistics(step_maximum, step_list)

        # prepare for selection
        batch_concat = self.prepare_for_selection(batch_concat, reward_tensor_concat, reward_extra_info_dict_concat,\
            reward_mean_50p, reward_std_50p)
        
        # ------------------------Only Select Positive BufferSize Samples------------------------------ #
        # here we only use the 2**N samples as trajectory buffer
        len_batch = batch_concat.batch["advantages"].size(0)
        len_batch_valid = nearest_lower_power_of_2(len_batch)
        # len_batch_valid = largest_ratio_32(len_batch)
        len_batch_valid = min(len_batch_valid, self.buffer_size)
        print("After Adv calculation in TrajectoryBuffer", f"{len_batch=}", f"{len_batch_valid=}")
        assert(len_batch_valid <= len_batch)
        mask = torch.zeros(len_batch, dtype=torch.bool)
        idx = torch.randperm(len_batch)[:len_batch_valid]
        mask[idx] = True
        # ------------------------Only Select Positive BufferSize Samples------------------------------ #
        
        batch_concat.batch = batch_concat.batch[mask]
        reward_tensor_concat = reward_tensor_concat[mask]
        # assert (len(batch_concat.batch) == len_batch_valid)
        # assert (len(reward_tensor_concat) == len_batch_valid)
        for key, value in batch_concat.non_tensor_batch.items():
            if isinstance(value, np.ndarray):
                batch_concat.non_tensor_batch[key] = value[mask]
            elif isinstance(value, list) or isinstance(value, tuple):
                batch_concat.non_tensor_batch[key] = [v for v, m in zip(value, mask) if m]
            else:
                batch_concat.non_tensor_batch[key] = value
        for key, value in batch_concat.meta_info.items():
            if isinstance(value, np.ndarray):
                batch_concat.meta_info[key] = value[mask]
            elif isinstance(value, list) or isinstance(value, tuple):
                batch_concat.meta_info[key] = [v for v, m in zip(value, mask) if m]
            else:
                batch_concat.meta_info[key] = value
        for key, value in reward_extra_info_dict_concat.items():
            if isinstance(value, np.ndarray):
                reward_extra_info_dict_concat[key] = value[mask]
            elif isinstance(value, list) or isinstance(value, tuple):
                reward_extra_info_dict_concat[key] = [v for v, m in zip(value, mask) if m]
            else:
                reward_extra_info_dict_concat[key] = value

        batch_concat.batch["reward_tensor"] = reward_tensor_concat
        batch_concat.meta_info["reward_extra_infos_dict"] = reward_extra_info_dict_concat
        batch_concat.meta_info["reward_mean_95p"] = reward_mean_95p
        batch_concat.meta_info["reward_std_95p"] = reward_std_95p
        batch_concat.meta_info["reward_mean_50p"] = reward_mean_50p
        batch_concat.meta_info["reward_std_50p"] = reward_std_50p
        return batch_concat


    def reset_buffer(self):
        self.trajectory_buffer = []


    def update_batch(self, batch_messages, data, data_reward_tensor, data_reward_extra_infos_dict, group_size, global_step):
        # 直接存储DataProto中advantage 正确的部分
        """
        data.batch is TensorDict with keys [\'advantages\', \'attention_mask\', \'input_ids\', \'old_log_probs\',
        \'position_ids\', \'prompts\', \'response_mask\', \'responses\', \'returns\',
        \'token_level_rewards\', \'token_level_scores\']'
        """
        batch = deepcopy(data)
        reward_tensor = deepcopy(data_reward_tensor)
        reward_extra_infos_dict = deepcopy(data_reward_extra_infos_dict)
        rm_scores = reward_tensor.sum(dim=-1)

        uid_list = batch.non_tensor_batch["uid"]
        traj_uid_list = batch.non_tensor_batch["traj_uid"]
        reward_by_uid_traj_uid = {}     
        idx2uid = {}
        idx = 0
        
        for uid, traj_uid, rm_score in zip(uid_list, traj_uid_list, rm_scores):
            idx2uid[idx] = uid
            if not (uid in reward_by_uid_traj_uid):
                reward_by_uid_traj_uid[uid] = {}
            if not (traj_uid in reward_by_uid_traj_uid[uid]):
                reward_by_uid_traj_uid[uid][traj_uid] = []            
            reward_by_uid_traj_uid[uid][traj_uid].append(rm_score.item())
            idx += 1

        # 这里计算每个样本每个轨迹的平均奖励 然后再计算整个样本组的平均奖励
        reward_by_uid_mean = []
        reward_by_uid_std = []
        for uid, reward_traj in reward_by_uid_traj_uid.items():
            reward_traj_rwscore = []
            for traj_uid in reward_traj.keys():
                reward_traj_rwscore.append(np.mean(reward_traj[traj_uid]))
            reward_by_uid_mean.append([uid, float(np.mean(reward_traj_rwscore))])
            reward_by_uid_std.append([uid, float(np.std(reward_traj_rwscore))])
        
        # import pdb;pdb.set_trace();
        in_group_std_list = [item[1] for item in reward_by_uid_std]
        in_group_mean_list = [item[1] for item in reward_by_uid_mean]

        ## update the baseline buffer consistently
        num_group_mean_all = sum([len(item[1]) for item in self.reward_mean_history])
        while num_group_mean_all > self.baseline_buffer_size and len(self.reward_mean_history):
            self.reward_mean_history.pop(0)
            self.reward_std_history.pop(0)
            num_group_mean_all = sum([len(item[1]) for item in self.reward_mean_history])
        self.reward_mean_history.append([global_step, in_group_mean_list])
        self.reward_std_history.append([global_step, in_group_std_list])

        # if self.weight_decay_trajectory_replay <= 0 or self.weight_decay_trajectory_replay > 1:
        #     # 如果该系数 <= 0 or > 1 则代表的是依靠移动平均reward来计算的
        #     self.trajectory_buffer.append([batch, reward_tensor, reward_extra_infos_dict, global_step])
        #     print("Add into the trajectory buffer directly", len(self.trajectory_buffer))
        #     return

        # 筛选出advantage > 0的样本
        mask = (batch.batch["advantages"].mean(-1) > 0)
        # 筛选出工具调用的样本
        # for batch_message_idx, batch_message in enumerate(batch_messages):
        #     # [不一定]必须要包含工具调用
        #     has_tool_call = False
        #     messages = batch_message["messages"]
        #     try:
        #         for message in messages:
        #             if "tool_calls" in message and message["tool_calls"] is not None:
        #                 has_tool_call = True
        #             assert("role" in message)
        #             assert("content" in message)
        #     except Exception as e:
        #         print("Found errors in messages", e)
        #         print(f"❌ dame Messages:\n\n{messages}")
        #         mask[batch_message_idx] = False
        #     if not has_tool_call:
        #         mask[batch_message_idx] = False

        batch.batch = batch.batch[mask]
        reward_tensor = reward_tensor[mask]
        print(">>> Updated batch into buffer before filtering ", len(mask), " after filtering ", len(reward_tensor))
        if len(reward_tensor) == 0:
            return

        for key, value in batch.non_tensor_batch.items():
            if isinstance(value, np.ndarray):
                batch.non_tensor_batch[key] = value[mask]
            elif isinstance(value, list) or isinstance(value, tuple):
                batch.non_tensor_batch[key] = [v for v, m in zip(value, mask) if m]
            else:
                batch.non_tensor_batch[key] = value

        for key, value in batch.meta_info.items():
            if isinstance(value, np.ndarray):
                batch.meta_info[key] = value[mask]
            elif isinstance(value, list) or isinstance(value, tuple):
                batch.meta_info[key] = [v for v, m in zip(value, mask) if m]
            else:
                batch.meta_info[key] = value

        for key, value in reward_extra_infos_dict.items():
            if isinstance(value, np.ndarray):
                reward_extra_infos_dict[key] = value[mask]
            elif isinstance(value, list) or isinstance(value, tuple):
                reward_extra_infos_dict[key] = [v for v, m in zip(value, mask) if m]
            else:
                reward_extra_infos_dict[key] = value

        # # 删除可能缺失的non_tensor_batch
        # non_tensor_batch_keys = list(batch.non_tensor_batch.keys())
        # for non_tensor_batch_key in non_tensor_batch_keys:
        #     if "_success_rate" in non_tensor_batch_key:
        #         del batch.non_tensor_batch[non_tensor_batch_key]
        self.trajectory_buffer.append([batch, reward_tensor, reward_extra_infos_dict, global_step])
        return


    def get_buffer_size(self):
        num_buffer = 0
        if self.weight_decay_trajectory_replay <= 0 or self.weight_decay_trajectory_replay > 1:
            if len(self.reward_mean_history):
                reward_mean_history = []
                for reward_item in self.reward_mean_history:
                    reward_mean_history += reward_item[1]
                reward_mean_50p = np.percentile(reward_mean_history, 50)
            else:
                reward_mean_50p = None
            for traj_pair in self.trajectory_buffer:
                _, reward_tensor, _, _ = traj_pair
                reward_tensor_sum = (reward_tensor.view(reward_tensor.size(0), -1)).sum(dim=-1)
                if reward_mean_50p is None:
                    reward_tensor_sum_pos = (reward_tensor_sum > reward_tensor_sum.mean())
                else:
                    reward_tensor_sum_pos = (reward_tensor_sum > reward_mean_50p)
                num_buffer += reward_tensor_sum_pos.sum()
        else:
            for traj_pair in self.trajectory_buffer:
                traj_batch, _, _, _ = traj_pair
                num_buffer += traj_batch.batch.size(0)
        return num_buffer



class RayPPOSFTTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def _build_dataloader_sft_online(self, train_dataset):
        # build dataset
        config = self.config
        self.train_dataset_sft_online = train_dataset
        # TODO: 这里发现一个小bug，训练应该把整个数据集都放进去迭代
        self.train_sampler_sft_online = DistributedSampler(self.train_dataset_sft_online, shuffle=True, num_replicas=1, rank=0, drop_last=False)
        self.train_dataloader_sft_online = DataLoader(
            dataset=self.train_dataset_sft_online,
            batch_size=len(self.train_dataset_sft_online),
            sampler=self.train_sampler_sft_online,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )



    def generate_batch(self, timing_raw, batch, gen_batch, metrics):
        with _timer("gen", timing_raw):
            # if not self.async_rollout_mode:
            #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            # else:
            #     self.async_rollout_manager.wake_up()
            #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
            #     self.async_rollout_manager.sleep()

            ################ agent-environment loop ###############
            gen_batch_output = self.traj_collector.multi_turn_loop(
                                                    gen_batch=gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.envs,
                                                    is_train=True,
                                                    )
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            with _timer("gen_max", timing_raw):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["do_sample"] = False
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                batch = batch.union(gen_baseline_output)
                reward_baseline_tensor = self.reward_fn(batch)
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                batch.batch["reward_baselines"] = reward_baseline_tensor

                del gen_baseline_batch, gen_baseline_output

        del batch
        batch = gen_batch_output

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
            step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                batch=batch,
                gamma=self.config.algorithm.gamma
            )
            batch.batch['step_rewards'] = step_rewards_tensor
        
        batch = adjust_batch(self.config, batch)
        batch.meta_info["global_steps"] = self.global_steps
        batch.meta_info["use_toolcall_reward"] = self.config.algorithm.use_toolcall_reward
        batch.meta_info["max_toolcall_steps"] = self.config.algorithm.max_toolcall_steps
        batch.batch["response_mask"] = compute_response_mask(batch)
        # balance the number of valid tokens on each dp rank.
        # Note that this breaks the order of data inside the batch.
        # Please take care when you implement group based adv computation such as GRPO and rloo
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        batch.meta_info["global_steps"] = self.global_steps

        return batch, gen_batch_output, metrics


    def recompute_old_log_prob_ref_critic(self, timing_raw, batch, metrics):
        # recompute old_log_probs
        with _timer("old_log_prob", timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

            if "rollout_log_probs" in batch.batch.keys():
                # TODO: we may want to add diff of probs too.
                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                actor_old_log_probs = batch.batch["old_log_probs"]
                attention_mask = batch.batch["attention_mask"]
                responses = batch.batch["responses"]
                response_length = responses.size(1)
                response_mask = attention_mask[:, -response_length:]

                rollout_probs = torch.exp(rollout_old_log_probs)
                actor_probs = torch.exp(actor_old_log_probs)
                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                metrics.update(
                    {
                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                    }
                )

        if self.use_reference_policy:
            # compute reference log_prob
            with _timer("ref", timing_raw):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # compute values
        if self.use_critic:
            with _timer("values", timing_raw):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)
        return timing_raw, batch, metrics


    def compute_advantage(self, timing_raw, batch, metrics,\
        future_reward, reward_tensor, reward_extra_infos_dict):
        with _timer("adv", timing_raw):
            # we combine with rule-based rm
            reward_extra_infos_dict: dict[str, list]
            if self.config.reward_model.launch_reward_fn_async:
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            batch.batch["token_level_scores"] = reward_tensor

            print(f"{list(reward_extra_infos_dict.keys())=}")
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            # compute rewards. apply_invalid_action_penalty if available
            if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                        invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                        )
                metrics.update(invalid_metrics)

            # compute rewards. apply_kl_penalty if available
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # compute advantages, executed on the driver process

            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                gigpo_mode=self.config.algorithm.gigpo.mode,
            )
        return timing_raw, batch, metrics,\
            future_reward, reward_tensor, reward_extra_infos_dict


    def log_rollout(self, timing_raw, batch, reward_extra_infos_dict):
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            with _timer("dump_rollout_generations", timing_raw):
                print(batch.batch.keys())
                inputs_ids_wo_padding = [[token_id for token_id in ids if token_id != self.tokenizer.pad_token_id] for ids in batch.batch["prompts"]]
                outputs_ids_wo_padding = [[token_id for token_id in ids if token_id != self.tokenizer.pad_token_id] for ids in batch.batch["responses"]]
                inputs = self.tokenizer.batch_decode(inputs_ids_wo_padding, skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(outputs_ids_wo_padding, skip_special_tokens=False)
                # inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                # outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                # ground_truths = [reward_model["ground_truth"] for reward_model in batch.non_tensor_batch["reward_model"]]
                ground_truths = batch.non_tensor_batch["anchor_obs"]
                assert(len(ground_truths) == len(inputs))
                self._dump_generations(
                    inputs=inputs,
                    outputs=outputs,
                    scores=scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    ground_truth_scores=ground_truths,
                    dump_path=rollout_data_dir,
                )
        return timing_raw, batch


    def validation_stepwise(self, timing_raw, metrics):
        with _timer("testing", timing_raw):
            val_metrics: dict = self._validate()
            val_steps = self.global_steps
            print(f"Step {val_steps} validation metrics: {val_metrics}")
            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir:
                val_jsonl_path = os.path.join(val_data_dir, f"validation_metrics_step-{val_steps}.jsonl")
                val_metrics_save = {}
                for k,v in val_metrics.items():
                    val_metrics_save[k] = float(v)
                with open(val_jsonl_path, "w") as fw:
                    fw.write(json.dumps(val_metrics_save, ensure_ascii=False, indent=4))
            if is_last_step:
                last_val_metrics = val_metrics
        metrics.update(val_metrics)
        return timing_raw, metrics


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir:
                val_steps = self.global_steps
                val_jsonl_path = os.path.join(val_data_dir, f"validation_metrics_step-{val_steps}.jsonl")
                val_metrics_save = {}
                for k, v in val_metrics.items():
                    val_metrics_save[k] = float(v)
                with open(val_jsonl_path, "w") as fw:
                    fw.write(json.dumps(val_metrics_save, ensure_ascii=False, indent=4))
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        
        ################ Newly Added Buffer Replay for Self-Imitation Learning PG Loss ################
        # here we respectively treat the trajectory buffer for replaying
        
        if self.config.actor_rollout_ref.actor.enable_trajectory_replay:
            trajectory_buffer_replay = TrajectoryBufferBatch(
                tokenizer=self.tokenizer,\
                    tolerate_steps=self.config.actor_rollout_ref.actor.trajectory_tolerate_steps,\
                        buffer_size=self.config.actor_rollout_ref.actor.trajectory_buffer_size,\
                            baseline_buffer_size=self.config.actor_rollout_ref.actor.baseline_buffer_size,\
                                weight_decay_trajectory_replay=self.config.actor_rollout_ref.actor.weight_decay_trajectory_replay,\
                                    adv_estimator=self.config.algorithm.adv_estimator,\
                                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True))
        else:
            trajectory_buffer_replay = None
        
        self.train_dataset_sft_online = None
        self.train_dataloader_sft_online = None
        self.train_dataset_replay_online = None
        # replay buffer set does not need dataloader but directly the dataproto itself
        ################ Newly Added Buffer Replay for SFT or PG Loss ################

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                gen_batch.meta_info["global_steps"] = self.global_steps
                is_last_step = self.global_steps >= self.total_training_steps
                
                with _timer("step", timing_raw):
                    # generate a batch
                    batch, gen_batch_output, metrics = self.generate_batch(timing_raw, batch, gen_batch, metrics)

                    # ################ Start of the Newly Added Buffer Replay for SFT or PG Loss ################
                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer, max_response_len=self.config.data.max_response_length)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn, max_response_len=self.config.data.max_response_length)
                            # some post-processing to filter out easily failed samples
                            if "is_incomplete" in reward_extra_infos_dict and self.config.algorithm.filter_incomplete_responses:
                                is_incomplete_list = reward_extra_infos_dict["is_incomplete"]
                                response_masks = deepcopy(batch.batch["response_mask"])
                                mask = torch.tensor(is_incomplete_list, dtype=torch.bool, device=response_masks.device)
                                mask_len = len(torch.nonzero(mask))
                                response_masks[mask] = 0   # 1 for llm generated tokens; 0 for tool observations
                                batch.batch["response_mask"] = response_masks
                                print(f"🔧 >>>>> Filter Out {mask_len} Incomplete Responses from Loss")
                                metrics.update(
                                    {
                                        "incomplete_responses": mask_len,
                                    }
                                )

                            if "is_overlong" in reward_extra_infos_dict and self.config.algorithm.filter_overlong_responses:
                                is_overlong_list = reward_extra_infos_dict["is_overlong"]
                                response_masks = deepcopy(batch.batch["response_mask"])
                                mask = torch.tensor(is_overlong_list, dtype=torch.bool, device=response_masks.device)
                                mask_len = len(torch.nonzero(mask))
                                response_masks[mask] = 0   # 1 for llm generated tokens; 0 for tool observations
                                batch.batch["response_mask"] = response_masks
                                print(f"🔧 >>>>> Filter Out {mask_len} Overlong Responses from Loss")
                                metrics.update(
                                    {
                                        "overlong_responses": mask_len,
                                    }
                                )
                            
                            if "is_repetitive" in reward_extra_infos_dict and self.config.algorithm.filter_repetitive_responses:
                                is_repetitive_list = reward_extra_infos_dict["is_repetitive"]
                                response_masks = deepcopy(batch.batch["response_mask"])
                                mask = torch.tensor(is_repetitive_list, dtype=torch.bool, device=response_masks.device)
                                mask_len = len(torch.nonzero(mask))
                                response_masks[mask] = 0   # 1 for llm generated tokens; 0 for tool observations
                                batch.batch["response_mask"] = response_masks
                                print(f"🔧 >>>>> Filter Out {mask_len} Repetitive Responses from Loss")
                                metrics.update(
                                    {
                                        "repetitive_responses": mask_len,
                                    }
                                )

                            if "is_unreadable" in reward_extra_infos_dict and self.config.algorithm.filter_unreadable_responses:
                                is_unreadable_list = reward_extra_infos_dict["is_unreadable"]
                                response_masks = deepcopy(batch.batch["response_mask"])
                                mask = torch.tensor(is_unreadable_list, dtype=torch.bool, device=response_masks.device)
                                mask_len = len(torch.nonzero(mask))
                                response_masks[mask] = 0   # 1 for llm generated tokens; 0 for tool observations
                                batch.batch["response_mask"] = response_masks
                                print(f"🔧 >>>>> Filter Out {mask_len} Unreadable BPE Token strs from Loss")
                                metrics.update(
                                    {
                                        "unreadableBPE_responses": mask_len,
                                    }
                                )

                            # here we check if we should filter out low variance groups
                            batch, metrics_update, reward_tensor, reward_extra_infos_dict = self._filter_rollout(batch, reward_tensor, reward_extra_infos_dict)
                            metrics.update(metrics_update)
                    # ################ End of the Newly Added Buffer Replay for SFT or PG Loss ################

                    # recompute old_log_probs
                    timing_raw, batch, metrics = self.recompute_old_log_prob_ref_critic(timing_raw, batch, metrics)

                    # advantage
                    timing_raw, batch, metrics, future_reward, reward_tensor, reward_extra_infos_dict =\
                        self.compute_advantage(timing_raw, batch, metrics, future_reward,\
                            reward_tensor, reward_extra_infos_dict)


                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        ################ Start of the Newly Added Buffer Replay for SFT or PG Loss ################
                        # 计算完loss后在这里进行训练更新梯度
                        # here we respectively treat the trajectory buffer for sft or replaying
                        if trajectory_buffer_replay is not None and trajectory_buffer_replay.is_full_capacity():
                            print(">>> ⚠️ REPLAY 通过设置bs=整个buffer大小实现1次loss计算结束")
                            batch_replay = trajectory_buffer_replay.get_buffer()
                            if len(batch_replay.batch) < self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes:
                                # 防止出现空的batch replay任务
                                print(">>>> 🚨 遇到了筛选adv后过少的replay buffer batch", len(batch_replay.batch))
                                batch_replay: DataProto = DataProto.from_single_dict({})
                            trajectory_buffer_replay.reset_buffer()
                        else:
                            batch_replay: DataProto = DataProto.from_single_dict({})
                        ################ End of the Newly Added Buffer Replay for SFT or PG Loss ################
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch, batch_replay)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                    ################ Start of the Newly Added Buffer Replay for SFT or PG Loss ################
                    # Check if the trajectory is correct and if so add into the buffer
                    scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    # import pdb;pdb.set_trace();
                    messages = self.convert_messages_format(batch.non_tensor_batch["messages"])
                    response_mask_list = batch.batch["response_mask"].sum(-1).cpu().tolist()
                    assert(len(scores) == len(messages))
                    assert(len(scores) == len(response_mask_list))
                    if trajectory_buffer_replay is not None:
                        # remove non-latest
                        trajectory_buffer_replay.remove_old()
                        # add in new messages
                        # 所有的其他内容应该都在batch里
                        trajectory_buffer_replay.update_batch(messages, batch, reward_tensor, reward_extra_infos_dict,\
                            self.config.actor_rollout_ref.rollout.n, self.global_steps)
                        trajectory_buffer_replay_buffersize = trajectory_buffer_replay.get_buffer_size()
                        print(">>> 📚 Current trajectory replay buffer size: ", trajectory_buffer_replay_buffersize)
                        metrics.update(
                            {
                                "trajectory_sft_buffersize": trajectory_buffer_replay_buffersize,
                            }
                        )
                    ################ End of the Newly Added Buffer Replay for SFT or PG Loss ################


                    # Log rollout generations if enabled
                    timing_raw, batch = self.log_rollout(timing_raw, batch, reward_extra_infos_dict)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        timing_raw, metrics = self.validation_stepwise(timing_raw, metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

