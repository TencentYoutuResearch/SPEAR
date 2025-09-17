import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import re
from ftlangdetect import detect
from collections import Counter
import numpy as np


def has_repeated_ngrams(words, n=20, threshold=10):
    # words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = Counter(ngrams)
    return any(count >= threshold for count in counts.values())


def remove_tool_tags(text):
    # 移除 <tool_call>...</tool_call>
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    # 移除 <tool_response>...</tool_response>
    text = re.sub(r'<tool_response>.*?</tool_response>', '', text, flags=re.DOTALL)
    return text


def default_compute_score_enforce_toolcall_posneg_decay(data_source, solution_str, ground_truth, user_query, file_path, tools=None, messages=[],\
    extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, global_steps=-1, use_toolcall_reward="none", max_toolcall_steps=200):
    """
    positive-negative: both contribute to the tool-call reward by the number of toolcall turns
    """
    res = default_compute_score(data_source, solution_str, ground_truth, user_query, file_path, tools=tools, messages=messages,\
        extra_info=extra_info, sandbox_fusion_url=sandbox_fusion_url, concurrent_semaphore=concurrent_semaphore, global_steps=global_steps)    
    res_toolcall = 0.0
    if tools is not None:
        for message in messages:
            if type(message) is dict:
                if message["role"] == "assistant" and ("tool_calls" in message and message["tool_calls"] is not None):
                    res_toolcall += 0.1
            else:
                if message.role == "assistant" and (message.tool_calls is not None):
                    res_toolcall += 0.1
    
    if res_toolcall == 0:
        print("tools", tools, "messages", messages)
        print(f"⚠️🔧 warning: the assistant message does not contain any tool calls")
    else:
        print(f"✅🔧 check tool calls OK")

    res_toolcall = min(res_toolcall, 1.0)   # number of toolcall maximum 1.0 reward
    assert(global_steps >= 0)
    if use_toolcall_reward == "none":
        res_toolcall_ratio = 0.0
    elif use_toolcall_reward == "constant":
        res_toolcall_ratio = 1
    elif use_toolcall_reward == "cosine":
        res_toolcall_ratio = 1
        if global_steps <= max_toolcall_steps:
            res_toolcall_ratio *= (np.cos((global_steps / max_toolcall_steps) * np.pi) + 1)/2
        else:
            res_toolcall_ratio = 0
    else:
        raise NotImplementedError("use_toolcall_reward must be one of ['none', 'constant', 'cosine']")

    if isinstance(res, dict):
        assert('score' in res)
        if "is_incomplete" in res and (res_toolcall) and (not res["is_incomplete"]):
            # 如果调用了工具但是 没有输出最终答案 那么仍然保留这个负样本
            # 如果既没有调用工具 也没有输出最终答案 那么肯定有问题（重复输出、退化）那么就直接把这个loss mask掉->is_incomplete=1
            res["is_incomplete"] = 0
        # plus the toolcall turn
        res['score'] += float(res_toolcall) * res_toolcall_ratio
        return res
    else:
        # plus the toolcall turn
        res += float(res_toolcall) * res_toolcall_ratio
        return res



def default_compute_score_enforce_toolcall_posneg_decay_qwen3(data_source, solution_str, ground_truth, user_query, file_path, tools=None, messages=[],\
    extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, global_steps=-1, use_toolcall_reward="none", max_toolcall_steps=200):
    """
    positive-negative: both contribute to the tool-call reward by the number of toolcall turns
    """
    res = default_compute_score(data_source, solution_str, ground_truth, user_query, file_path, tools=tools, messages=messages,\
        extra_info=extra_info, sandbox_fusion_url=sandbox_fusion_url, concurrent_semaphore=concurrent_semaphore, global_steps=global_steps)    
    res_toolcall = 0.0
    if tools is not None:
        for message in messages:
            if type(message) is dict:
                if message["role"] == "assistant" and ("tool_calls" in message and message["tool_calls"] is not None):
                    res_toolcall += 0.1
            else:
                if message.role == "assistant" and (message.tool_calls is not None):
                    res_toolcall += 0.1
    
    if res_toolcall == 0:
        print("tools", tools, "messages", messages)
        print(f"⚠️🔧 warning: the assistant message does not contain any tool calls")
    else:
        print(f"✅🔧 check tool calls OK")

    res_toolcall = min(res_toolcall, 1.0)   # number of toolcall maximum 1.0 reward
    assert(global_steps >= 0)
    if use_toolcall_reward == "none":
        res_toolcall_ratio = 0.0
    elif use_toolcall_reward == "constant":
        res_toolcall_ratio = 1
    elif use_toolcall_reward == "cosine":
        res_toolcall_ratio = 1
        if global_steps <= max_toolcall_steps:
            res_toolcall_ratio *= (np.cos((global_steps / max_toolcall_steps) * np.pi) + 1)/2
        else:
            res_toolcall_ratio = 0
    else:
        raise NotImplementedError("use_toolcall_reward must be one of ['none', 'constant', 'cosine']")

    res_toolcall_ratio = max(res_toolcall_ratio, 0)
    if isinstance(res, dict):
        assert('score' in res)
        if "is_incomplete" in res and (res_toolcall) and (not res["is_incomplete"]):
            # 如果调用了工具但是 没有输出最终答案 那么仍然保留这个负样本
            # 如果既没有调用工具 也没有输出最终答案 那么肯定有问题（重复输出、退化）那么就直接把这个loss mask掉->is_incomplete=1
            res["is_incomplete"] = 0
        # plus the toolcall turn
        if res['score'] <= 0:
            res['score'] += float(res_toolcall) * res_toolcall_ratio
        else:
            res['score'] += float(res_toolcall > 0) * res_toolcall_ratio
        return res
    else:
        # plus the toolcall turn
        if res <= 0:
            res += float(res_toolcall) * res_toolcall_ratio
        else:
            res += float(res_toolcall > 0) * res_toolcall_ratio

        return res



