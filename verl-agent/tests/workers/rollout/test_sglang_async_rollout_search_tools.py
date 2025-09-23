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
from utils_sglang import (
    get_rollout_config,
    prepare_inputs,
)

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
    # """
    # Generate test conversation data for search tool integration scenarios.
    
    # This function creates the core test data structure that simulates a multi-turn 
    # conversation between a user and an AI assistant with search tool capabilities.
    # It defines the complete conversation flow used across all search tool tests.
    
    # Conversation Flow:
    #     1. User Query: "How's the weather lately?" (with reasoning instructions)
    #     2. Assistant Turn 0: Searches for "today's weather" via tool call
    #     3. Tool Response 0: Returns current weather information  
    #     4. Assistant Turn 1: Searches for "tomorrow's weather" via tool call
    #     5. Tool Response 1: Returns future weather information
    #     6. Assistant Turn 2: Provides final answer combining both search results
    
    # Data Structure:
    #     - user_prompts: List containing the initial user query with search instructions
    #     - expect_turn_array: Expected assistant responses at each conversation turn
    #     - tool_return_array: Mock tool responses for each search query
    
    # Search Tool Integration:
    #     - Uses OpenAI function calling format with "search" tool
    #     - Each tool call includes properly formatted arguments with query parameters
    #     - Tool responses simulate realistic search engine results
    #     - Final assistant response demonstrates information synthesis
    
    # Test Coverage:
    #     - Multi-turn conversation handling
    #     - Tool call parsing and execution
    #     - Search result integration into responses  
    #     - Proper conversation state management
    #     - OpenAI function calling schema compliance
    
    # Returns:
    #     tuple: (user_prompts, expect_turn_array, tool_return_array)
    #         - user_prompts: Initial user messages for conversation start
    #         - expect_turn_array: Expected assistant responses for each turn
    #         - tool_return_array: Mock search tool response messages
    
    # Note:
    #     This function provides the foundation data for all search tool tests,
    #     ensuring consistent conversation patterns and expected behaviors across
    #     different test scenarios (basic, batch, error handling, etc.).
    # """
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

    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_tools_registration(
        self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config
    ):
        rollout = SGLangRollout(
            actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config
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
        # """
        # Test the creation and preprocessing of async rollout requests with search tool capabilities.
        
        # This test validates the core request preprocessing functionality that converts
        # input DataProto objects into AsyncRolloutRequest instances with proper search
        # tool integration and OpenAI function calling schema validation.
        
        # Key Validation Areas:
        #     - Request list creation from DataProto input
        #     - Initial request state (PENDING) assignment
        #     - Search tool schema registration and validation
        #     - OpenAI function calling format compliance
        #     - Tool parameter schema correctness
        
        # Request Creation Process:
        #     1. SGLangRollout initialization with search tool configuration
        #     2. DataProto preprocessing into AsyncRolloutRequest objects
        #     3. Tool schema extraction and validation
        #     4. Request state initialization and metadata setup
        
        # Schema Validation:
        #     - Tool type: "function" for OpenAI compatibility
        #     - Function name: "search" matching tool configuration
        #     - Description: Web search capability description
        #     - Parameters: Proper OpenAI function parameters schema
        #     - Required fields: "query_list" for search queries
        #     - Property types: Array of strings for query inputs
        
        # Expected Request Properties:
        #     - Single request created from input DataProto
        #     - Initial state: PENDING (ready for processing)
        #     - Tool schemas: 1 search tool schema registered
        #     - Schema format: Full OpenAI function calling specification
        #     - Parameter validation: Required query_list with proper typing
        
        # Integration Points:
        #     - SearchTool integration through rollout configuration
        #     - OpenAI function calling schema generation
        #     - Request preprocessing pipeline validation
        #     - Tool metadata propagation from DataProto
        
        # Args:
        #     search_rollout_config: Configuration with search tool enabled
        #     qwen_tokenizer: Tokenizer for request processing
        #     qwen_model_config: Model configuration
        #     search_data_proto: Input data containing search metadata
        # """
        rollout = SGLangRollout(
            actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config
        )
        req_list = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)
        assert len(req_list) == 1
        assert req_list[0].state == AsyncRolloutRequestStateEnum.PENDING
        assert len(req_list[0].tools) == 1
        print(type(req_list[0].tools[0]))
        assert req_list[0].tools[0] == OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="search",
                description="Searches the web for relevant information based on the given query.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query_list": OpenAIFunctionPropertySchema(
                            type="array",
                            description="A list of fully-formed semantic queries. The tool will return search results for each query.",
                            items={"type": "string"},
                        )
                    },
                    required=["query_list"],
                ),
                strict=False,
            ),
        )

    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_over_size_case(
        self,
        mock_env,
        mock_engine,
        mock_sampling,
        search_rollout_config,
        qwen_tokenizer,
        qwen_model_config,
        search_data_proto,
        search_data,
    ):
        # """
        # Test handling of response truncation due to length limits in search tool scenarios.
        
        # This test validates the rollout system's behavior when responses are truncated
        # due to maximum length constraints, specifically testing the edge case where
        # the assistant's response is cut off before tool calls can be made.
        
        # Test Scenario:
        #     - Artificially limit max_turns to 1 to force early termination
        #     - Mock engine response with "length" finish_reason indicating truncation
        #     - Verify proper handling when conversation is cut short
        #     - Ensure graceful degradation without tool execution
        
        # Key Behaviors Tested:
        #     - Response truncation handling with "length" finish_reason
        #     - Request state management under constraint conditions
        #     - Proper completion without tool call execution
        #     - Empty reward scores when no tools are executed
        #     - Message count validation for truncated conversations
        
        # Mock Configuration:
        #     - max_turns: Set to 1 to force early termination
        #     - finish_reason: "length" with length=3000 to simulate truncation
        #     - Engine response: First turn only, no tool calls processed
        #     - Tool execution: Bypassed due to early termination
        
        # Expected Behavior:
        #     - Request completes with COMPLETED state despite truncation
        #     - No search tool rewards collected (empty reward_scores)
        #     - Only 2 messages total: user prompt + truncated assistant response
        #     - Assistant response matches expected first turn content
        #     - No tool_calls in final assistant message
        
        # Validation Points:
        #     - Proper state transition under length constraints
        #     - Correct message sequence for truncated conversations
        #     - Empty metrics collection when tools aren't executed
        #     - Graceful handling of incomplete multi-turn scenarios
        #     - No errors or exceptions during truncation scenarios
        
        # Args:
        #     search_rollout_config: Configuration with modified max_turns limit
        #     qwen_tokenizer: Tokenizer for message processing
        #     qwen_model_config: Model configuration
        #     search_data_proto: Input request data
        #     search_data: Expected conversation patterns (only first turn used)
        # """
        search_rollout_config.multi_turn.max_turns = 1
        rollout = SGLangRollout(
            actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config
        )
        req = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        _, expect_turn_array, _ = search_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        rollout._handle_engine_call = MagicMock()
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
        rollout._handle_engine_call.return_value = future
        rollout._tp_rank = 0
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[rollout._async_rollout_a_request(req, True, False) for req in req_list],
            )
        )
        assert len(output_req_list) == 1
        output_req = output_req_list[0]
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        assert output_req.reward_scores == {"search": []}, f"output_req.reward_scores: {output_req.reward_scores}"
        # we should only have two message, one for prompt, second for response.
        assert len(output_req.messages) == 2
        assert output_req.messages[1] == Message(
            role="assistant",
            content=expect_turn_array[0],
            tool_calls=None,
        )

    @patch.object(SearchTool, "execute", new_callable=AsyncMock)
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_tool_call_basic_case(
        self,
        mock_sampling,
        mock_engine,
        mock_env,
        mock_execute,
        search_rollout_config,
        qwen_tokenizer,
        qwen_model_config,
        search_data_proto,
        search_data,
    ):
        _, expect_turn_array, tool_return_array = search_data

        # Mock search tool execution to return predefined responses
        mock_execute.side_effect = [(msg, 0.0, {"status": "success"}) for msg in tool_return_array]

        search_rollout_config.multi_turn.max_turns = 10
        rollout = SGLangRollout(
            actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config
        )

        rollout._tool_map["search"].retrieval_service_url = "mock://dummy"

        req = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        rollout._handle_engine_call = MagicMock()
        futures = [asyncio.Future() for i in expect_turn_array]
        for idx, (i, turn) in enumerate(zip(futures, expect_turn_array, strict=False)):
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
                assert rollout._function_call_parser.has_tool_call(turn)
                assert rollout._function_call_parser.parse_non_stream(turn)

        rollout._handle_engine_call.side_effect = futures
        rollout._tp_rank = 0

        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(*[rollout._async_rollout_a_request(req, True, False) for req in req_list])
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
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_tool_call_batch_case(
        self,
        mock_sampling,
        mock_engine,
        mock_env,
        mock_execute,
        search_rollout_config,
        qwen_tokenizer,
        qwen_model_config,
        search_data_proto,
        search_data,
    ):
        _, expect_turn_array, tool_return_array = search_data

        # Mock tool execution for large batch (100 requests * 2 calls each)
        mock_execute.side_effect = [
            (tool_return_array[0], 0.0, {"status": "success"}),
            (tool_return_array[1], 0.0, {"status": "success"}),
        ] * 100

        search_rollout_config.multi_turn.max_turns = 10
        rollout = SGLangRollout(
            actor_module="",
            config=search_rollout_config,
            tokenizer=qwen_tokenizer,
            model_hf_config=qwen_model_config,
        )
        rollout._tool_map["search"].retrieval_service_url = "mock://dummy"

        base_req = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]

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
            for idx, (fut, turn) in enumerate(zip(futures, expect_turn_array, strict=False)):
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
            rollout._tp_rank = 0
            loop = asyncio.get_event_loop()
            output_req_list = loop.run_until_complete(
                asyncio.gather(*[rollout._async_rollout_a_request(r, True, False) for r in req_list])
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
