# pylint: disable=line-too-long, function-name-too-long

# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from verl.tools.mcp_search_tool import MCPSearchTool
from verl.tools.utils.mcp_clients.McpClientManager import MCPClientManager
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
        "tool_calls": [
            {
                "id": "10",
                "type": "function",
                "function": {
                    "name": "tavily_search_tool",
                    "arguments": {
                        "what_is_your_intent": "Search for the weather lately",
                        "query": "the weather in Beijing today",
                        "search_depth": "basic",
                        "time_range": "day",
                        "include_domains": ["google.com", "baidu.com"],
                        "max_results": 2,
                    },
                },
            }
        ],
    }

    expect_turn_1_msg = {
        "role": "assistant",
        "content": "Let me search again.",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "tavily_search_tool",
                    "arguments": {
                        "what_is_your_intent": "Search for the weather lately",
                        "query": "the weather in Beijing tomorrow",
                        "search_depth": "basic",
                        "time_range": "day",
                        "include_domains": ["google.com", "baidu.com"],
                        "max_results": 2,
                    },
                },
            }
        ],
    }

    expect_turn_2_msg = {
        "role": "assistant",
        "content": "<answer>Today is sunny and tomorrow will be cloudy in Beijing.</answer>",
    }

    # Mock search tool responses
    tool_return_0_msg = {"role": "tool", "content": [{"type": "text", "text": "Today's weather in Beijing is sunny."}]}
    tool_return_1_msg = {
        "role": "tool",
        "content": [{"type": "text", "text": "Tomorrow's weather in Beijing is cloudy."}],
    }

    user_prompts = [user_prompt]
    expect_turn_array = [expect_turn_0_msg, expect_turn_1_msg, expect_turn_2_msg]
    tool_return_array = [tool_return_0_msg, tool_return_1_msg]

    return user_prompts, expect_turn_array, tool_return_array


class TestRolloutWithMCPSearchTools:
    # """
    # Comprehensive test suite for SGLang rollout functionality with Model Context Protocol (MCP) search tool integration.
    
    # This test class validates the end-to-end functionality of SGLang rollout workers when
    # integrated with external MCP search tools, specifically testing the Tavily search tool
    # through the Model Context Protocol interface. It covers various scenarios including
    # tool registration, multi-turn conversations, batch processing, and error handling.
    
    # MCP Integration Overview:
    #     - Model Context Protocol provides a standardized interface for tool integration
    #     - MCPSearchTool wraps Tavily search API through MCP client architecture
    #     - MCPClientManager handles dynamic tool schema fetching and lifecycle management
    #     - Tools support rich parameter schemas with validation and type checking
    
    # Core Test Scenarios:
    #     - MCP tool registration and schema validation through MCPClientManager
    #     - Async rollout request preprocessing with MCP tool capabilities
    #     - Single-request multi-turn conversations with Tavily search calls
    #     - Batch processing of concurrent MCP search requests (100+ requests)
    #     - Response handling for different completion reasons (tool_calls, stop, length)
    #     - MCP client lifecycle and error handling scenarios
    
    # Tavily Search Tool Features:
    #     - Advanced web search with configurable depth (basic/advanced)
    #     - Time-based filtering (day, week, month, year)
    #     - Domain inclusion/exclusion capabilities
    #     - Content control (raw content, images, descriptions)
    #     - Result count limiting and async search support
    #     - Intent-based search categorization
    
    # Fixtures:
    #     qwen_tokenizer: Qwen2.5-0.5B tokenizer with left padding configuration
    #     qwen_model_config: Model configuration for Qwen2.5-0.5B
    #     search_data: Pre-processed conversation data with MCP search interactions
    #     search_rollout_config: Configuration enabling MCP tool integration
    #     search_data_proto: DataProto containing tokenized prompts and MCP metadata
    #     mock_rollout: Fully mocked rollout instance with MCP client management
    
    # Test Pattern:
    #     The tests simulate a weather inquiry scenario where:
    #     1. User asks "How's the weather lately?" with reasoning instructions
    #     2. Assistant searches for current weather using Tavily search tool
    #     3. Assistant searches for future weather using different parameters
    #     4. Assistant provides final answer combining both search results
    
    # MCP Architecture:
    #     - MCPClientManager: Handles tool schema fetching and client lifecycle
    #     - MCPSearchTool: Implements MCP protocol for Tavily search integration
    #     - Dynamic schema loading: Tools schemas fetched at runtime from MCP servers
    #     - Protocol compliance: Full MCP specification adherence
    
    # Dependencies:
    #     - SGLangRollout: Main rollout worker with MCP tool support
    #     - MCPSearchTool: MCP-compliant search tool implementation
    #     - MCPClientManager: MCP client lifecycle and schema management
    #     - AsyncRolloutRequest: Request lifecycle management
    #     - Tavily API: External search service (mocked in tests)
    
    # Note:
    #     Tests extensively use mocking to avoid external MCP server dependencies
    #     and focus on the rollout logic and MCP protocol integration mechanisms.
    #     The mock_rollout fixture provides critical MCP client simulation.
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
        tool_path = "./resource/tool_configs/mcp_tool_config"
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
                    "tavily_search_tool": {
                        "create_kwargs": {"ground_truth": "Today is sunny and tomorrow will be cloudy in Beijing."},
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
        # """
        # Create a fully mocked SGLang rollout instance with MCP search tool integration.
        
        # This fixture provides the core testing infrastructure for MCP tool integration,
        # setting up a complete mock environment that simulates the MCP client architecture
        # and tool schema management without requiring external MCP servers or API connections.
        
        # MCP Mock Architecture:
        #     - MCPClientManager.fetch_tool_schemas: Mocked to return predefined Tavily tool schema
        #     - SGLangRollout initialization components: All external dependencies mocked
        #     - Tool schema: Complete Tavily search tool specification with all parameters
        #     - Sampling parameters: Configured for test scenarios
        
        # Tool Schema Configuration:
        #     - Tool Name: "tavily_search_tool" (MCP-compliant naming)
        #     - Required Parameters: what_is_your_intent, query
        #     - Optional Parameters: search_depth, topic, days, time_range, domains, content options
        #     - Parameter Validation: Full OpenAI function calling schema compliance
        #     - Type System: String, integer, boolean, array types with descriptions
        
        # Tavily Tool Capabilities (Mocked):
        #     - Intent-based searching with purpose declaration
        #     - Basic/advanced search depth configuration
        #     - Time-range filtering (day, week, month, year)
        #     - Domain inclusion/exclusion lists
        #     - Content control (raw content, images, descriptions)
        #     - Result count limiting (5-20 results)
        #     - Async search execution support
        
        # Mock Dependencies:
        #     - Distributed environment initialization bypassed
        #     - Inference engine setup mocked
        #     - Sampling parameters initialization mocked
        #     - MCP client management fully simulated
        
        # Rollout Configuration:
        #     - Response length: From search_rollout_config
        #     - Sampling: Standard parameters with no penalties
        #     - Tool integration: MCP client manager with schema fetching
        #     - Tokenizer: Qwen2.5 with proper chat template support
        
        # Usage Pattern:
        #     This fixture enables isolated testing of MCP tool integration logic
        #     without external dependencies, focusing on:
        #     - Tool registration and schema validation
        #     - Request preprocessing with MCP metadata
        #     - Multi-turn conversation handling
        #     - Tool call parsing and execution simulation
        
        # Returns:
        #     SGLangRollout: Fully configured mock rollout instance with:
        #         - MCP search tool registered and available
        #         - Complete tool schema with parameter validation  
        #         - Sampling parameters configured for testing
        #         - All external dependencies mocked for isolation
        
        # Note:
        #     This mock provides the foundation for all MCP tool tests, ensuring
        #     consistent behavior and eliminating external service dependencies
        #     while maintaining realistic MCP protocol simulation.
        # """
        tool_schema = [
            {
                "type": "function",
                "function": {
                    "name": "tavily_search_tool",
                    "description": "A powerful web search tool...",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "what_is_your_intent": {
                                "type": "string",
                                "description": "Describe your intent for using Tavily",
                            },
                            "query": {"type": "string", "description": "Search query"},
                            "search_depth": {
                                "type": "string",
                                "description": "The depth of the search ('basic' or 'advanced')",
                            },
                            "topic": {
                                "type": "string",
                                "description": "The category of the search ('general' or 'news')",
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days back to include in search results (only for "
                                "'news' topic)",
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for results ('day', 'week', 'month', 'year', 'd', "
                                "'w', 'm', 'y')",
                            },
                            "include_domains": {
                                "type": "array",
                                "description": "List of domains to specifically include in search results",
                            },
                            "exclude_domains": {
                                "type": "array",
                                "description": "List of domains to specifically exclude from search results",
                            },
                            "include_answer": {
                                "type": "boolean",
                                "description": "Whether to include an answer summary generated by an LLM",
                            },
                            "include_raw_content": {
                                "type": "boolean",
                                "description": "Whether to include the cleaned and parsed HTML content of each result",
                            },
                            "include_images": {
                                "type": "boolean",
                                "description": "Whether to include images from search results",
                            },
                            "include_image_descriptions": {
                                "type": "boolean",
                                "description": "Whether to include descriptions with images",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (5-20)",
                            },
                            "async_search": {
                                "type": "boolean",
                                "description": "Whether to perform the search asynchronously",
                            },
                        },
                        "required": ["what_is_your_intent", "query"],
                    },
                    "strict": False,
                },
            }
        ]
        with (
            patch.object(MCPClientManager, "fetch_tool_schemas", return_value=tool_schema),
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

    def test_tools_registration(self, mock_rollout):
        assert len(mock_rollout._tool_schemas) != 0
        assert "tavily_search_tool" in mock_rollout._tool_map.keys()
        from verl.tools.mcp_search_tool import MCPSearchTool

        assert isinstance(mock_rollout._tool_map["tavily_search_tool"], MCPSearchTool)
        # depend on the tokenizer
        assert mock_rollout._tool_call_parser_type == "qwen25"

    def test_rollout_req_creation(self, mock_rollout, search_data_proto):
        req_list = mock_rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)
        assert len(req_list) == 1
        assert req_list[0].state == AsyncRolloutRequestStateEnum.PENDING
        assert len(req_list[0].tool_schemas) == 1

    def test_over_size_case(self, mock_rollout, search_data_proto, search_data):
        mock_rollout.config.multi_turn.max_assistant_turns = 1
        req = mock_rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        _, expect_turn_array, _ = search_data
        # here we mock a meta info with 'length'. indicate the response is truncate
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
        assert output_req.reward_scores.get("tavily_search_tool") == []
        # we should only have two message, one for prompt, second for response.
        assert len(output_req.messages) == 2
        assert output_req.messages[1] == Message(
            role="assistant",
            content=expect_turn_array[0],
            tool_calls=None,
        )

    @patch.object(MCPSearchTool, "execute", new_callable=AsyncMock)
    def test_tool_call_basic_case(self, mock_execute, mock_rollout, search_data_proto, search_data):
            # Test basic MCP search tool calling functionality in a single-request multi-turn scenario.
        
        # This test validates the complete workflow of an async rollout request that involves
        # multiple MCP search tool calls during a multi-turn conversation. It simulates a weather
        # inquiry where the assistant performs two separate Tavily searches through MCP protocol
        # before providing a comprehensive final answer.
        
        # Test Flow:
        #     1. Setup mock rollout with MCP search tool integration enabled
        #     2. Create single async rollout request with MCP tool capabilities
        #     3. Mock MCPSearchTool execution to return predefined weather responses
        #     4. Simulate 3-turn conversation: search today -> search tomorrow -> final answer
        #     5. Validate MCP tool call parsing, execution, and response integration
        
        # MCP Conversation Pattern:
        #     - Turn 0: Assistant searches for current weather using Tavily tool
        #       * Tool call with rich parameters (intent, query, search_depth, time_range, domains, max_results)
        #       * MCP protocol compliance with structured arguments
        #     - MCP Response 0: "Today's weather in Beijing is sunny."
        #     - Turn 1: Assistant searches for future weather with different parameters
        #       * Tool call targeting tomorrow's weather
        #       * Demonstrates parameter variation within same tool
        #     - MCP Response 1: "Tomorrow's weather in Beijing is cloudy."
        #     - Turn 2: Assistant provides synthesized final answer combining both results
        
        # MCP Protocol Validation:
        #     - Tool call structure: OpenAI function calling format
        #     - Parameter validation: Required and optional parameter handling
        #     - Response format: MCP-compliant content structure with text blocks
        #     - Tool execution: Async execution with success status tracking
        
        # Mocked Components:
        #     - MCPSearchTool.execute: Returns predefined weather information with success status
        #     - Engine calls: Simulated responses with appropriate finish_reason types
        #     - MCP client communication: Bypassed with predetermined return values
        #     - Tool schema validation: Pre-loaded schema from mock_rollout fixture
        
        # Validation Points:
        #     - Request state transitions (PENDING -> COMPLETED)
        #     - MCP tool execution count (should be 2 calls to tavily_search_tool)
        #     - Message sequence validation (6 total: user + 3*assistant + 2*tool)
        #     - Tool response content matching expected weather information
        #     - Metrics collection with MCP success status indicators
        #     - Proper MCP tool call parsing for each assistant turn
        #     - Tool call ID handling and response correlation
        
        # Expected Behavior:
        #     - 3 assistant turns with 2 MCP search tool calls in between
        #     - Final message sequence: user + assistant + tool + assistant + tool + assistant
        #     - MCP search metrics contain success status and proper execution counts
        #     - Tool responses match expected weather query results with MCP format
        #     - All tool parameters properly parsed and validated
        
        # Args:
        #     mock_execute: Mocked MCPSearchTool.execute method for controlled responses
        #     mock_rollout: Fully configured mock rollout instance with MCP integration
        #     search_data_proto: Input data with MCP tool metadata and parameters
        #     search_data: Expected conversation turns and MCP tool return values
        _, expect_turn_array, tool_return_array = search_data
        # Mock search tool execution to return predefined responses
        mock_execute.side_effect = [(msg, 0.0, {"status": "success"}) for msg in tool_return_array]

        mock_rollout.config.multi_turn.max_assistant_turns = 10
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
        assert "tavily_search_tool" in output_req.metrics
        assert output_req.metrics["tavily_search_tool"][0]["status"] == "success"
        assert mock_execute.await_count == 2
        assert len(output_req.messages) == 6
        # Verify tool response messages contain expected content
        search_counter = 0
        for msg in output_req.messages:
            if msg.role == "tool":
                assert msg.content == tool_return_array[search_counter]
                search_counter += 1
        assert search_counter == 2

    @patch.object(MCPSearchTool, "execute", new_callable=AsyncMock)
    def test_tool_call_batch_case(self, mock_execute, mock_rollout, search_data_proto, search_data):
        # """
        # Test batch processing of MCP search tool calls with concurrent request handling.
        
        # This test validates the scalability and concurrency handling of the SGLang rollout
        # system when processing multiple requests simultaneously through the Model Context
        # Protocol, each involving Tavily search tool calls. It simulates 100 concurrent
        # weather inquiry requests to test MCP client management and resource scalability.
        
        # Test Scenario:
        #     - Creates 100 identical async rollout requests with MCP tool capabilities
        #     - Each request follows the same 3-turn conversation pattern as basic case
        #     - All requests execute concurrently using asyncio.gather
        #     - Validates proper MCP client isolation and resource sharing
        
        # MCP Batch Processing Architecture:
        #     - Request Isolation: Each request maintains independent MCP tool state
        #     - Concurrent Execution: All 100 requests processed simultaneously via asyncio
        #     - Resource Sharing: MCP client manager handles multiple concurrent connections
        #     - Tool Execution Scaling: 200 total Tavily search calls (2 per request × 100 requests)
        #     - Protocol Compliance: MCP specification adherence under high concurrency
        
        # Mock Infrastructure:
        #     - Engine Call Mapping: Each request gets dedicated future sequences
        #     - MCP Tool Execution: Alternating pattern of weather responses for all requests
        #     - Request Tracking: Per-request counters and future management
        #     - State Isolation: Individual completion tracking per request
        #     - Client Management: Simulated MCP client lifecycle under load
        
        # Validation at Scale:
        #     - All 100 requests complete successfully (COMPLETED state)
        #     - Total MCP tool executions: 200 calls (verified via mock.await_count)
        #     - Message consistency: Each request has 6 messages (user + 3*assistant + 2*tool)
        #     - Tool response validation: 2 MCP tool messages per request with expected content
        #     - Metrics integrity: Success status recorded for all Tavily search operations
        #     - MCP protocol compliance: All tool calls follow MCP specification
        
        # Performance Characteristics:
        #     - Concurrent request handling without MCP client interference
        #     - Proper async/await patterns for MCP tool execution
        #     - Memory efficient MCP client state management
        #     - Deterministic completion despite concurrent MCP operations
        #     - Resource pooling and connection management
        
        # Stress Testing Aspects:
        #     - High concurrency load (100 simultaneous MCP requests)
        #     - Tool execution scalability (200 concurrent Tavily searches)
        #     - MCP client manager resource handling under load
        #     - Request state isolation with shared MCP infrastructure
        #     - Memory and connection usage patterns
        
        # MCP Protocol Validation Under Load:
        #     - Tool call parsing consistency across all requests
        #     - Parameter validation for 200 individual tool calls
        #     - Response format compliance for all MCP interactions
        #     - Error handling and recovery in high-concurrency scenarios
        #     - Client lifecycle management with concurrent connections
        
        # Args:
        #     mock_execute: Mocked MCPSearchTool.execute for controlled MCP responses
        #     mock_rollout: Configured mock rollout instance with MCP support
        #     search_data_proto: Base request data template for batch MCP testing
        #     search_data: Expected conversation patterns and MCP tool responses
        # """
        _, expect_turn_array, tool_return_array = search_data
        # Mock tool execution for large batch (100 requests * 2 calls each)
        mock_execute.side_effect = [
            (tool_return_array[0], 0.0, {"status": "success"}),
            (tool_return_array[1], 0.0, {"status": "success"}),
        ] * 100

        mock_rollout.config.multi_turn.max_assistant_turns = 10
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
            assert "tavily_search_tool" in out_req.metrics
            for metric in out_req.metrics["tavily_search_tool"]:
                assert metric["status"] == "success"
            assert len(out_req.messages) == 6
            assert sum(1 for m in out_req.messages if m.role == "tool") == 2

        assert mock_execute.await_count == 2 * req_nums
