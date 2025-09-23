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

import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
import ray.util.multiprocessing

from verl.tools.base_tool import BaseTool
from verl.utils.reward_score.prime_code import compute_score as compute_score_prime
from verl.utils.reward_score.sandbox_fusion import compute_score as compute_score_sandbox

from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2

import sys

sys.set_int_max_str_digits(10000)

@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        # this only used for observalability
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class ExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        # TODO validation for rate_limit
        # A Singleton Rate Limitor
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        with ExitStack() as stack:
            stack.callback(self.rate_limit_worker.release.remote)
            ray.get(self.rate_limit_worker.acquire.remote())
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                # TODO we should make this available to the tool caller
                logger.warning(f"Error when executing code: {e}")


def init_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
    if mode == PoolMode.ThreadMode:
        return ray.remote(ExecutionWorker).options(max_concurrency=num_workers).remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
    else:
        raise NotImplementedError("Process mode is not implemented yet")
        # return ray.util.multiprocessing.Pool(processes=num_workers)



class SandBoxTool(BaseTool):
    """A demo tool for calculating the reward of code.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 4)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 60)
        # self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(num_workers=self.num_workers, enable_global_rate_limit=self.enable_global_rate_limit, rate_limit=self.rate_limit, mode=PoolMode.ThreadMode)
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        log_msg = f"🔥🔥🔥 Init SandBoxTool (For Coding) with config: {config}"
        logger.info(log_msg)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        if not isinstance(ground_truth, dict):
            try:
                test_cases = json.loads(ground_truth)
                # 考虑到
                # 1) inptus/outputs可能很多 拖累代码编译速度;
                # 2) 模型能看到全部input-output信息，可能泄露答案hardcode
                # 这里我们采用少量用于验证（至多3个）
                if type(test_cases) is dict and "inputs" in test_cases and "outputs" in test_cases:
                    test_cases["inputs"] = test_cases["inputs"][:3]
                    test_cases["outputs"] = test_cases["outputs"][:3]
                ground_truth_str = json.dumps(test_cases, ensure_ascii=False)
            except json.JSONDecodeError:
                ground_truth_str = ground_truth
        else:
            ground_truth_str = json.dumps(ground_truth, ensure_ascii=False)

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth_str,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], assistant_content="", time_limit=30, **kwargs) -> tuple[str, float, dict]:
        # code = parameters.get("code", "")
        code = str(assistant_content)
        self._instance_dict[instance_id]["response"] = code
        reward = 0.0
        msg = []
        reward, msg = await self.execution_pool.execute.remote(self.execute_code, instance_id, time_limit)
        msg_str = json.dumps(msg, ensure_ascii=False)
        # penalty for non improved answer submission
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        self._instance_dict[instance_id]["reward"] = reward
        print(f"<<< execution messages 💌 {msg_str=}, 🏆 {reward=}")
        return f"Code Execution Feedback\n\n{msg_str}\n\nReward={reward}", tool_reward, {}

    def execute_code(self, instance_id, timeout=60, concurrent_semaphore=None, continuous=True, **kwargs) -> float:
        if self.sandbox_fusion_url:
            res = compute_score_sandbox(self.sandbox_fusion_url, concurrent_semaphore,\
                self._instance_dict[instance_id]["response"],\
                self._instance_dict[instance_id]["ground_truth"],\
                continuous=continuous,
                timeout=timeout)

        else:
            res = compute_score_prime(
                self._instance_dict[instance_id]["response"],
                self._instance_dict[instance_id]["ground_truth"],\
                continuous=continuous,
                timeout=timeout)

        if isinstance(res, dict):
            return res, ""

        elif isinstance(res, (int, float, bool)):
            return float(res), ""
        else:
            return float(res[0]), res[1]

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

