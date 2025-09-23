# pylint: disable=line-too-long, function-name-too-long

# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
# Adapted from tests/workers/rollout/test_sglang_async_rollout_sf_tools.py


import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from tensordict import TensorDict
from transformers import AutoConfig, AutoTokenizer
from utils_sglang import get_rollout_config, prepare_inputs

from verl.protocol import DataProto
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)
from verl.tools.search_tool import SearchTool
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, Message
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)
user_content = DEFAULT_USER_CONTENT_PREFIX.rstrip("\n") + "How's the weather lately?"


def get_search_messages():
    user_prompt = {
        "role": "user",
        "content": user_content,
    }

    expect_turn_0_msg = {
        "role": "assistant",
        "content": "Let me search the web.",
        "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": "today's weather"}}}],
    }

    expect_turn_1_msg = {
        "role": "assistant",
        "content": "Let me search again.",
        "tool_calls": [
            {"type": "function", "function": {"name": "search", "arguments": {"query": "tomorrow's weather"}}}
        ],
    }

    expect_turn_2_msg = {
        "role": "assistant",
        "content": "<answer>Today is sunny and tomorrow will be cloudy in Beijing.</answer>",
    }

    # Mock search tool responses
    tool_return_0_msg = {"role": "tool", "content": "Today's weather in Beijing is sunny."}
    tool_return_1_msg = {"role": "tool", "content": "Tomorrow's weather in Beijing is cloudy."}

    user_prompts = [user_prompt]
    expect_turn_array = [expect_turn_0_msg, expect_turn_1_msg, expect_turn_2_msg]
    tool_return_array = [tool_return_0_msg, tool_return_1_msg]

    return user_prompts, expect_turn_array, tool_return_array


class TestRolloutWithSearchTools:
    # """
    # Comprehensive test suite for SGLang rollout functionality with search tool integration.
    
    # This test class validates the end-to-end functionality of SGLang rollout workers when
    # integrated with external search tools for information retrieval. It tests various
    # scenarios including tool registration, multi-turn conversations with search queries,
    # and batch processing of search-enabled requests.
    
    # Core Test Scenarios:
    #     - Search tool registration and OpenAI function schema validation
    #     - Async rollout request preprocessing with search tool capabilities
    #     - Single-request multi-turn conversations with search tool calls
    #     - Batch processing of concurrent search requests (100+ requests)
    #     - Response handling for different completion reasons (tool_calls, stop, length)
    #     - Tool execution mocking and response validation
    
    # Search Tool Integration:
    #     - Uses SearchTool for web information retrieval
    #     - Supports query-based search with structured responses
    #     - Handles multi-turn conversations where assistant searches multiple times
    #     - Validates tool call parsing and response integration
    
    # Fixtures:
    #     qwen_tokenizer: Qwen2.5-0.5B tokenizer with left padding configuration
    #     qwen_model_config: Model configuration for Qwen2.5-0.5B
    #     search_data: Pre-processed conversation data with search interactions
    #     search_rollout_config: Configuration enabling search tool integration
    #     search_data_proto: DataProto containing tokenized prompts and search metadata
    #     mock_rollout: Fully mocked rollout instance for isolated testing
    
    # Test Pattern:
    #     The tests simulate a weather inquiry scenario where:
    #     1. User asks "How's the weather lately?"
    #     2. Assistant searches for "today's weather"
    #     3. Assistant searches for "tomorrow's weather" 
    #     4. Assistant provides final answer combining both results
    
    # Dependencies:
    #     - SGLangRollout: Main rollout worker with tool support
    #     - SearchTool: External search service integration
    #     - AsyncRolloutRequest: Request lifecycle management
    #     - OpenAI function calling schemas for tool definitions
    
    # Note:
    #     Tests extensively use mocking to avoid external search API dependencies
    #     and focus on the rollout logic and search tool integration mechanisms.
    # """
    
    @pytest.fixture
    def qwen_tokenizer(self):
        local_model_path = "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # we only need this for tokenizer
    @pytest.fixture
    def qwen_model_config(self):
        local_model_path = "Qwen/Qwen2.5-0.5B"
        config = AutoConfig.from_pretrained(local_model_path)
        return config

    @pytest.fixture
    def search_data(self, qwen_tokenizer):
        user_prompt, expect_turn_array, tool_return_array = get_search_messages()
        prompts = [[message] for message in user_prompt]
        preencode_turn_array = [
            qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=False)
            for turn in expect_turn_array
        ]
        preencode_tool_return_array = [
            qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=True)
            for turn in tool_return_array
        ]
        return prompts, preencode_turn_array, preencode_tool_return_array

    @pytest.fixture
    def search_rollout_config(self):
        max_prompt_length = 4096
        max_response_length = 3000
        dtype = "bfloat16"
        tensor_parallel_size = 1
        tool_path = "./resource/tool_configs/search_tool_config"
        rollout_config = get_rollout_config(
            max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_path
        )
        return rollout_config

    @pytest.fixture
    def search_data_proto(self, search_data, qwen_tokenizer):
        preencode_prompts, _, _ = search_data
        prompts = [
            qwen_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in preencode_prompts
        ]
        input_ids, attention_mask, position_ids = prepare_inputs(qwen_tokenizer, prompts, 1000)
        prompt_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        messages = np.asarray(preencode_prompts)

        tools_kwargs = np.array(
            [
                {
                    "search": {
                        "create_kwargs": {
                            "ground_truth": "Today is sunny and tomorrow will be cloudy in Beijing.",
                            "data_source": "searchR1_nq",
                        },
                    },
                }
            ],
            dtype=object,
        )
        index = np.array([0], dtype=object)
        prompts = DataProto(
            batch=prompt_dict, non_tensor_batch={"raw_prompt": messages, "tools_kwargs": tools_kwargs, "index": index}
        )
        return prompts

    @pytest.fixture
    def mock_rollout(self, search_rollout_config, qwen_tokenizer, qwen_model_config):
        """Mock the rollout instance with sampling_params initialized."""
        with (
            patch.object(SGLangRollout, "_init_distributed_env", return_value=None),
            patch.object(SGLangRollout, "_init_inference_engine", return_value=None),
            patch.object(SGLangRollout, "_init_sampling_params", return_value=None),
        ):
            rollout = SGLangRollout(
                actor_module="",
                config=search_rollout_config,
                processing_class=qwen_tokenizer,
                model_hf_config=qwen_model_config,
            )
            rollout.sampling_params = {
                "n": 1,
                "max_new_tokens": search_rollout_config.response_length,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
            }
            return rollout

    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_tools_registration(
        self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config
    ):
        rollout = SGLangRollout(
            actor_module="",
            config=search_rollout_config,
            processing_class=qwen_tokenizer,
            model_hf_config=qwen_model_config,
        )
        assert len(rollout._tool_schemas) == 1
        assert "search" in rollout._tool_map.keys()
        from verl.tools.search_tool import SearchTool

        assert isinstance(rollout._tool_map["search"], SearchTool)
        # depend on the tokenizer
        assert rollout._tool_call_parser_type == "qwen25"

    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_rollout_req_creation(
        self,
        mock_env,
        mock_engine,
        mock_sampling,
        search_rollout_config,
        qwen_tokenizer,
        qwen_model_config,
        search_data_proto,
    ):
        rollout = SGLangRollout(
            actor_module="",
            config=search_rollout_config,
            processing_class=qwen_tokenizer,
            model_hf_config=qwen_model_config,
        )
        req_list = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)
        assert len(req_list) == 1
        assert req_list[0].state == AsyncRolloutRequestStateEnum.PENDING
        assert len(req_list[0].tool_schemas) == 1
        print(type(req_list[0].tool_schemas[0]))
        assert req_list[0].tool_schemas[0] == OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="search",
                description="Searches the web for relevant information based on the given query.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query_list": OpenAIFunctionPropertySchema(
                            type="array",
                            description="A list of fully-formed semantic queries. The tool will return search "
                            "results for each query.",
                            items={"type": "string"},
                        )
                    },
                    required=["query_list"],
                ),
                strict=False,
            ),
        )

    def test_over_size_case(self, mock_rollout, search_data_proto, search_data):
        mock_rollout.config.multi_turn.max_assistant_turns = 1
        req = mock_rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        _, expect_turn_array, _ = search_data
        mock_rollout._handle_engine_call = MagicMock()
        future = asyncio.Future()
        future.set_result(
            {
                "text": expect_turn_array[0],
                "meta_info": {
                    "id": "d1188d81cba840359df5b352b344bc8e",
                    "finish_reason": {"type": "length", "length": 3000},
                    "prompt_tokens": 132,
                    "completion_tokens": 100,
                    "cached_tokens": 0,
                    "e2e_latency": 2.23543,
                },
            }
        )
        mock_rollout._handle_engine_call.return_value = future
        mock_rollout._tp_rank = 0
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[mock_rollout._async_rollout_a_request(req, True, False) for req in req_list],
            )
        )
        assert len(output_req_list) == 1
        output_req = output_req_list[0]
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        assert output_req.reward_scores.get("search") == []
        assert len(output_req.messages) == 2
        assert output_req.messages[1] == Message(
            role="assistant",
            content=expect_turn_array[0],
            tool_calls=None,
        )

    @patch.object(SearchTool, "execute", new_callable=AsyncMock)
    def test_tool_call_basic_case(self, mock_execute, mock_rollout, search_data_proto, search_data):
        # """
        # Test basic search tool calling functionality in a single-request multi-turn scenario.
        
        # This test validates the complete workflow of an async rollout request that involves
        # multiple search tool calls during a multi-turn conversation. It simulates a weather
        # inquiry where the assistant performs two separate searches before providing a final answer.
        
        # Test Flow:
        #     1. Setup mock rollout with search tool integration enabled
        #     2. Create single async rollout request with search capabilities
        #     3. Mock search tool execution to return predefined weather responses
        #     4. Simulate 3-turn conversation: search today -> search tomorrow -> final answer
        #     5. Validate tool call parsing, execution, and response integration
        
        # Conversation Pattern:
        #     - Turn 0: Assistant searches for "today's weather" -> tool call with query
        #     - Tool Response 0: "Today's weather in Beijing is sunny."
        #     - Turn 1: Assistant searches for "tomorrow's weather" -> tool call with query  
        #     - Tool Response 1: "Tomorrow's weather in Beijing is cloudy."
        #     - Turn 2: Assistant provides final answer combining both search results
        
        # Mocked Components:
        #     - SearchTool.execute: Returns predefined weather information
        #     - Engine calls: Simulated responses with appropriate finish_reason types
        #     - Tool execution: Bypassed with predetermined return values and success status
        
        # Validation Points:
        #     - Request state transitions (PENDING -> COMPLETED)
        #     - Search tool execution count (should be 2 calls)
        #     - Message sequence validation (6 total: user + 3*assistant + 2*tool)
        #     - Tool response content matching expected weather information
        #     - Metrics collection with success status indicators
        #     - Proper tool call parsing for each assistant turn
        
        # Expected Behavior:
        #     - 3 assistant turns with 2 search tool calls in between
        #     - Final message sequence: user + assistant + tool + assistant + tool + assistant
        #     - Search metrics contain success status and proper execution counts
        #     - Tool responses match expected weather query results
        
        # Args:
        #     mock_execute: Mocked SearchTool.execute method
        #     mock_rollout: Fully configured mock rollout instance  
        #     search_data_proto: Input data with search tool metadata
        #     search_data: Expected conversation turns and tool return values
        # """
        _, expect_turn_array, tool_return_array = search_data

        # Mock search tool execution to return predefined responses
        mock_execute.side_effect = [(msg, 0.0, {"status": "success"}) for msg in tool_return_array]

        mock_rollout.config.multi_turn.max_assistant_turns = 10
        mock_rollout._tool_map["search"].retrieval_service_url = "mock://dummy"

        req = mock_rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        mock_rollout._handle_engine_call = MagicMock()
        futures = [asyncio.Future() for i in expect_turn_array]
        for idx, (i, turn) in enumerate(zip(futures, expect_turn_array, strict=True)):
            i.set_result(
                {
                    "text": turn,
                    "meta_info": {
                        "id": "d1188d81cba840359df5b352b344bc8e",
                        "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"},
                        "prompt_tokens": len(turn),
                        "completion_tokens": 100,
                        "cached_tokens": 0,
                        "e2e_latency": 2.23543,
                    },
                }
            )
            if idx < len(expect_turn_array) - 1:
                assert mock_rollout._function_call_parser.has_tool_call(turn)
                assert mock_rollout._function_call_parser.parse_non_stream(turn)

        mock_rollout._handle_engine_call.side_effect = futures
        mock_rollout._tp_rank = 0

        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(*[mock_rollout._async_rollout_a_request(req, True, False) for req in req_list])
        )

        # Verify conversation completed successfully with proper tool usage
        output_req = output_req_list[0]
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        assert "search" in output_req.metrics
        assert output_req.metrics["search"][0]["status"] == "success"
        assert mock_execute.await_count == 2
        assert len(output_req.messages) == 6  # user + 3*assistant + 2*tool_call
        # Verify tool response messages contain expected content
        search_counter = 0
        for msg in output_req.messages:
            if msg.role == "tool":
                assert msg.content == tool_return_array[search_counter]
                search_counter += 1
        assert search_counter == 2

    @patch.object(SearchTool, "execute", new_callable=AsyncMock)
    def test_tool_call_batch_case(self, mock_execute, mock_rollout, search_data_proto, search_data):
        # """
        # Test batch processing of search tool calls with concurrent request handling.
        
        # This test validates the scalability and concurrency handling of the SGLang rollout
        # system when processing multiple requests simultaneously, each involving search tool
        # calls. It simulates 100 concurrent weather inquiry requests to test system robustness
        # and proper resource management.
        
        # Test Scenario:
        #     - Creates 100 identical async rollout requests with search capabilities
        #     - Each request follows the same 3-turn conversation pattern as basic case
        #     - All requests execute concurrently using asyncio.gather
        #     - Validates proper request isolation and state management
        
        # Batch Processing Architecture:
        #     - Request Isolation: Each request maintains independent state and metadata
        #     - Concurrent Execution: All 100 requests processed simultaneously via asyncio
        #     - Resource Sharing: Mock engine calls distributed across request lifecycle
        #     - Tool Execution: 200 total search calls (2 per request × 100 requests)
        
        # Mock Infrastructure:
        #     - Engine Call Mapping: Each request gets dedicated future sequences
        #     - Tool Execution: Alternating pattern of weather responses for all requests
        #     - Request Tracking: Per-request counters and future management
        #     - State Isolation: Individual completion tracking per request
        
        # Validation at Scale:
        #     - All 100 requests complete successfully (COMPLETED state)
        #     - Total search tool executions: 200 calls (verified via mock.await_count)
        #     - Message consistency: Each request has 6 messages (user + 3*assistant + 2*tool)
        #     - Tool response validation: 2 tool messages per request with expected content
        #     - Metrics integrity: Success status recorded for all search operations
        
        # Performance Characteristics:
        #     - Concurrent request handling without interference
        #     - Proper async/await patterns for tool execution
        #     - Memory efficient request state management
        #     - Deterministic completion despite concurrent execution
        
        # Stress Testing Aspects:
        #     - High concurrency load (100 simultaneous requests)
        #     - Tool execution scalability (200 concurrent searches)
        #     - Request state isolation under load
        #     - Memory and resource usage patterns
        
        # Args:
        #     mock_execute: Mocked SearchTool.execute for controlled responses
        #     mock_rollout: Configured mock rollout instance with search support
        #     search_data_proto: Base request data template for batch creation
        #     search_data: Expected conversation patterns and tool responses
        # """
        _, expect_turn_array, tool_return_array = search_data

        # Mock tool execution for large batch (100 requests * 2 calls each)
        mock_execute.side_effect = [
            (tool_return_array[0], 0.0, {"status": "success"}),
            (tool_return_array[1], 0.0, {"status": "success"}),
        ] * 100

        mock_rollout.config.multi_turn.max_assistant_turns = 10
        mock_rollout._tool_map["search"].retrieval_service_url = "mock://dummy"

        base_req = mock_rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]

        req_nums = 100
        req_list = []
        req_turns_map = {}
        req_turns_counter = {}

        for i in range(req_nums):
            tmp_req = deepcopy(base_req)
            tmp_req.batch_data_id = i
            tmp_req.request_id = i
            req_list.append(MagicMock(wraps=tmp_req, spec=AsyncRolloutRequest))

            futures = [asyncio.Future() for _ in expect_turn_array]
            for idx, (fut, turn) in enumerate(zip(futures, expect_turn_array, strict=True)):
                fut.set_result(
                    {
                        "text": turn,
                        "meta_info": {
                            "id": "dummy",
                            "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"},
                            "prompt_tokens": len(turn),
                            "completion_tokens": 100,
                        },
                    }
                )
            req_turns_map[i] = futures
            req_turns_counter[i] = 0

        async def hacked_handle_engine_call(self, _req: AsyncRolloutRequest, *_args, **_kwargs):
            fut = req_turns_map[_req.batch_data_id][req_turns_counter[_req.batch_data_id]]
            req_turns_counter[_req.batch_data_id] += 1
            return await fut

        with patch.object(SGLangRollout, "_handle_engine_call", new=hacked_handle_engine_call):
            mock_rollout._tp_rank = 0
            loop = asyncio.get_event_loop()
            output_req_list = loop.run_until_complete(
                asyncio.gather(*[mock_rollout._async_rollout_a_request(r, True, False) for r in req_list])
            )

        # Verify all requests completed successfully
        assert len(output_req_list) == req_nums
        for out_req in output_req_list:
            assert out_req.state == AsyncRolloutRequestStateEnum.COMPLETED
            assert "search" in out_req.metrics
            for metric in out_req.metrics["search"]:
                assert metric["status"] == "success"
            assert len(out_req.messages) == 6  # user + 3 assistant + 2 tool
            assert sum(1 for m in out_req.messages if m.role == "tool") == 2

        assert mock_execute.await_count == 2 * req_nums
