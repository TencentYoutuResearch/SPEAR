# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import re
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
import ray.util.multiprocessing

from verl.tools.base_tool import BaseTool
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case

from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

import sys

sys.set_int_max_str_digits(10000)


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


WRAPPER_CODE = """
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
import numpy as np

"""


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


def remove_code_fence(code):
    # 正则说明：
    # ^```[\w\+\-]*\n  匹配以```开头，后面跟0个或多个字母、数字、下划线、+、-，再跟一个换行
    # (.*?)            非贪婪匹配中间的内容
    # \n```$           匹配以换行和```结尾
    pattern = r"^```[\w\+\-]*\n(.*?)\n```$"
    match = re.match(pattern, code, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return code  # 如果不匹配，原样返回


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


def init_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(ExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")
        # return ray.util.multiprocessing.Pool(processes=num_workers)


class SandboxFusionTool(BaseTool):
    """A tool for executing the code using sanbox fusion image.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A tool for execute code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "code needs to be execute and grad",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        # TODO: better documentation for the config
        self.num_workers = config.get("num_workers", 4)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.default_memory_limit_mb = config.get("memory_limit_mb", 1024)
        self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        log_msg = f"🔥🔥🔥 Init SandboxFusionTool (For Maths) with config: {config}"
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
            "reward": [],
        }
        return instance_id

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], assistant_content="", time_limit=30, **kwargs
    ) -> tuple[str, float, dict]:
        timeout = time_limit
        # timeout = parameters.get("timeout", self.default_timeout)
        memory_limit_mb = parameters.get("memory_limit_mb", self.default_memory_limit_mb)
        language = parameters.get("language", self.default_language)
        answer = str(assistant_content)
        if not isinstance(answer, str):
            answer = str(answer)
        code = parameters.get("code", "")
        if code == "":
            return "Empty input argument. You should pass your code into the function arguments.", 0, {}
        else:
            msg = ""

        code = remove_code_fence(code)
        code = WRAPPER_CODE + code.strip()  ## here add wrapper
        result = await self.execution_pool.execute.remote(
            self.execute_code, instance_id, code, timeout, memory_limit_mb, language
        )
        # result = self.execute_code(instance_id, code, timeout, language)
        print(f"<<< execution messages 💌 {result=}")
        # return result, result, result.strip()
        return msg + result, 0, {}

    def execute_code(self, instance_id, code, timeout=30, memory_limit_mb=1024, language="python"):
        result_status, metadata = _process_single_case(
            0, None, None, self.sandbox_fusion_url, code, timeout, memory_limit_mb=memory_limit_mb, language=language
        )
        # we should always expect this since we don't have correct answer
        if "case_index" in metadata:
            metadata.pop("case_index")
        if "input" in metadata:
            metadata.pop("input")
        if "expected_output" in metadata:
            metadata.pop("expected_output")
        if "api_response" in metadata:
            metadata.pop("api_response")
        if "status" in metadata:
            metadata.pop("status")

        metadata_str = json.dumps(metadata, ensure_ascii=False)
        if metadata["run_status"] == "Finished":
            std_error = metadata["stderr"] if metadata["stderr"] is not None else ""
            std_output = metadata["stdout"] if metadata["stdout"] is not None else ""
            if len(std_error) != 0:
                # Perhaps the tool response error should be clear
                std_error = str(std_error)
                std_error_split = std_error.split("\n")
                std_error_split = [item.strip() for item in std_error_split if len(item.strip()) > 0]
                std_error = "\n".join(std_error_split[-2:])
                msg = f"{std_error}"
                msg += "\nErrors occurred! Check your code."
            else:
                msg = f"{std_output}"
                if len(std_output) == 0:
                    msg += "\nEmpty stdout! You might forget to print the answer."
        else:
            msg = "Tool API failed! Please try again."
        return msg

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
