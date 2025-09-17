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

import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

# from verl.utils.reward_score import gsm8k
from verl.utils.reward_score import math_verify
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MathsAnswerTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        tools:
        - class_name: "verl.tools.maths_tool.MathsAnswerTool"
            config: {}
            tool_schema:
            type: "function"
            function:
                name: "calc_answer_reward"
                description: "A tool for calculating the reward of the answer to a maths question (1.0 if the parsed answer is correct, 0.0 if the parsed answer is incorrect or parsed incorrectly)."
                parameters:
                type: "object"
                properties:
                    answer:
                    type: "string"
                    description: "The final answer to the math problem wrapped by \\boxed{}."
                required: ["answer"]
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
        answer = answer.strip()

        if answer.startswith("\\boxed{"):
            self._instance_dict[instance_id]["response"] = answer
            answer_content = (answer[answer.find("\\boxed{")+len("\\boxed{"):answer.rfind("}")]).strip()
        else:
            answer_content = answer.strip()
            self._instance_dict[instance_id]["response"] = "\\boxed{" + answer + "}"
        # print("Maths Answer to be compared:", self._instance_dict[instance_id]["response"], " with ground-truth", self._instance_dict[instance_id]["ground_truth"])
        reward = await self.calc_reward(instance_id)
        # penalty for non improved answer submission
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        self._instance_dict[instance_id]["reward"] = reward
        return f"Current parsed {answer_content=} {reward=}", tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return math_verify.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"])

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
