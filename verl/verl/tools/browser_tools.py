import logging
import traceback

# import contextlib
from typing import Any, Optional
from uuid import uuid4

from .base_tool import BaseTool
from .browser_mcp_client import BrowserToolkitClient
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger("browser_toolkit")


class BrowserTool(BaseTool):
    """
    A tool should support the following methods:
    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        # self._exit_stack = contextlib.AsyncExitStack()

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        # mcp_client = await self._exit_stack.enter_async_context(BrowserToolkitClient(self.config["mcp_url"]))
        mcp_client = BrowserToolkitClient(self.config["mcp_url"])
        await mcp_client._start()
        self._instance_dict[instance_id] = {
            "mcp_client": mcp_client,
            "actions": [],
        }
        return instance_id

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], download_path: Optional[str] = None, **kwargs
    ) -> tuple[str, float, dict]:
        """Execute the tool.
        Args:
            download_path: `_download_path` for MCP server
        Returns: tool_response, tool_reward_score, tool_metrics
        """
        try:
            assert isinstance(parameters, dict), f"parameters must be a dict, got {type(parameters)}"
            assert "name" in parameters, (
                f"parameters must contain 'name', got {parameters}"
            )  # tools like `go_back` has not params
            action_name, action_params = parameters["name"], parameters.get("params", {})
            if download_path:
                action_params["_download_path"] = str(download_path)
            mcp_client: BrowserToolkitClient = self._instance_dict[instance_id]["mcp_client"]
            result = await mcp_client.call_tool(action_name, action_params)
            msg_action, msg_state = result.content[0].text, result.content[1].text
            msg = f"Action result: {msg_action}\n\n>>>>> Page Content\nNOTE that the following is one-time information!\n{msg_state}"
            self._instance_dict[instance_id]["actions"].append(
                {"name": action_name, "params": action_params, "result": msg_action}
            )
            return msg, 1.0, {}
        except Exception as e:
            logger.error(f"Error in execute: {instance_id} - `{parameters}`  \n{e}\n{traceback.format_exc()}")
            return f"Error in execute tool: {e}\n{traceback.format_exc()}", 0.0, {"error": str(e)}

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        try:
            mcp_client: BrowserToolkitClient = self._instance_dict[instance_id]["mcp_client"]
            await mcp_client._stop()
        except Exception as e:
            logger.error(f"Error in release: {e}\n{traceback.format_exc()}")
        finally:
            del self._instance_dict[instance_id]
