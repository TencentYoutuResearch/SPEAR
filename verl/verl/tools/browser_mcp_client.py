""" 
Basic client for MCP server. See [examples] for usage.

- [x] add _start() before any action to avoid manual call async function when initialization
- [x] support async context manager for automatic start/stop

updated: @2025-06-13
"""
import logging
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("browser_toolkit")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("browser_toolkit.log")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BrowserToolkitClient:
    """Browser toolkit client with async context manager support for automatic start/stop.
    
    Usage:
        async with BrowserToolkitClient(mcp_url) as client:
            result = await client.call_tool("tool_name", {"arg": "value"})
    """
    session: ClientSession | None = None
    session_id: str | None = None
    _exit_stack: AsyncExitStack | None = None

    def __init__(self, mcp_url: str) -> None:
        self.mcp_url = mcp_url

    async def _start(self) -> None:
        if self._exit_stack is not None:
            return
        logger.info("Starting BrowserToolkitClient...")
        self._exit_stack = AsyncExitStack()
        (read_stream, write_stream, get_session_id) = await self._exit_stack.enter_async_context(
            streamablehttp_client(
                url=self.mcp_url,
                timeout=timedelta(seconds=60*5),
                sse_read_timeout=timedelta(seconds=60*5),
                terminate_on_close=True,
            )
        )
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()
        # NOTE: session_id is initialized after session initialization!
        self.session_id = get_session_id()
        logger.info(f"[{self.session_id}] BrowserToolkitClient started")

    async def _stop(self) -> None:
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning(f"[{self.session_id}] Error in _stop: {e}, passing!")
            self._exit_stack = None
            self.session = None
            self.session_id = None
            logger.info(f"[{self.session_id}] BrowserToolkitClient stopped")

    async def list_tools(self) -> types.ListToolsResult:
        await self._start()
        return await self.session.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, read_timeout_seconds: timedelta | None = None) -> types.CallToolResult:
        await self._start()
        res = await self.session.call_tool(name, arguments, read_timeout_seconds)
        if res.isError:
            logger.error(f"[{self.session_id}] Error in call_tool: {res}")
        else:
            logger.info(f"[{self.session_id}] call_tool: `{name}({arguments})` -> {str(res.content)[:50]}")
        return res

    # DONE: expose the download information? or, identify by session_id? -- letter one

    async def __aenter__(self):
        await self._start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self._stop()
        except Exception as e:
            logger.error(f"[{self.session_id}] Error in __aexit__: {e}")

