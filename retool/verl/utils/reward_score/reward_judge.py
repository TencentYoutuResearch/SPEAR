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
import traceback
from copy import deepcopy
# from .utils import check_correctness
from openai import OpenAI

"""
Verify code correctness using the Sandbox Fusion (https://github.com/bytedance/SandboxFusion).
You can either deploy the sandbox_fusion service yourself or use the
FaaS service provided by public cloud, eg: volcengine.com.
"""
logger = logging.getLogger(__name__)

# reward_model_url = "https://ms-zgsm2lrf-100034032793-sw.gw.ap-shanghai.ti.tencentcs.com/ms-zgsm2lrf/v1"  ## 外网调用地址
# reward_model_url = "http://172.17.0.215/ms-zgsm2lrf/v1"  ## 内网调用地址
# reward_model_name = "qwen2.5-7B-Instruct"

# reward_model_url = "https://ms-lls6ssmj-100034032793-sw.gw.ap-shanghai.ti.tencentcs.com/ms-lls6ssmj/v1"  ## 外网调用地址
reward_model_url = "http://172.17.0.215/ms-lls6ssmj/v1" ## 内网调用地址
global reward_model_name
reward_model_name = "Qwen3-8B-Instruct"
global client
client = OpenAI(base_url=reward_model_url, api_key="xxx", timeout=60)


EVALUATION_PROMPT="""You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""



def compute_score(solution_str, ground_truth, user_query, timeout=10):
    """Compute the reward score for questions with reference responses
    """
    solution = deepcopy(solution_str)
    if "Final Answer:" in solution:
        solution = solution[solution.rfind("Final Answer:")+len("Final Answer:"):]

    else:
        return 0.0, "Invalid response format (missing 'Final Answer: <your final answer>')"

    prompt = EVALUATION_PROMPT.replace("{query}", user_query).replace("{result}", solution).replace("{answer}", ground_truth)
    global client
    global reward_model_name
    score = 0.0
    try:
        num_retry = 0
        while num_retry < 10:
            try:
                messages = [
                    {"role":"user", "content":prompt}
                ]                
                generation_kwargs = {
                    "max_tokens":16,
                    "temperature":0,
                    "stream":False,
                }
                response = client.with_options(timeout=timeout).chat.completions.create(
                                            model=reward_model_name,  
                                            messages=messages,
                                            **generation_kwargs
                                            )
                
                response = response.to_dict()
                assert "choices" in response
                result = response["choices"][0]["message"]["content"]
                assert (result is not None)
                result = str(result)
                if "CORRECT" in result.upper():
                    score = 1.0
                else:
                    score = 0.0
                return float(score)

            except Exception as e:
                print(f"{num_retry=} Inference error occurred", e)
                num_retry += 1
                continue
        raise ValueError("num_retry exceeds maximum of 10 for reward inference")

    except Exception as e:
        logger.error(f"Error during remote reward modeling compute_score: {e}")
        traceback.print_exc()
        score = 0.0

    # Ensure float and list are returned
    return float(score)


if __name__ == "__main__":
    
    user_query = "Who is the current chair of the VALSE常务AC委员会（LACC）?"
    ground_truth = "白翔（华中科技大学）"
    # solution_str = "白翔老师"
    solution_str = "Final Answer: 白翔老师"
    print(compute_score(solution_str, ground_truth, user_query, timeout=10))
