import json
import asyncio
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack
from isek.util.logger import logger

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from dotenv import load_dotenv
from isek.agent.tools.base import MCPTool

class MCPHandler:
    """
    Handles all MCP-related functionality including server connection,
    tool discovery, and tool execution.
    """
    def __init__(self) -> None:
        """
        Initialize the MCP handler.
        """
        self.logger = logger
        
        # MCP components
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI()
        
        # Load environment variables
        load_dotenv()

    async def connect_to_server(self, server_script_path: str) -> None:
        """
        Connect to an MCP server.

        :param server_script_path: Path to the server script (.py or .js)
        """
        try:
            is_python = server_script_path.endswith('.py')
            is_js = server_script_path.endswith('.js')
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command,
                args=[server_script_path],
                env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            await self.session.initialize()
            self.logger.info("Connected to MCP server")
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def get_available_tools(self) -> List[MCPTool]:
        """
        Get all available tools from the MCP server.

        :return: List of MCPTool instances
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            response = await self.session.list_tools()
            mcp_tools = []
            
            for tool in response.tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description,
                    mcp_client=self,  # Pass self as the MCP client
                    parameter_properties=tool.inputSchema.get("properties", {}),
                    required_parameters=tool.inputSchema.get("required", [])
                )
                mcp_tools.append(mcp_tool)
            
            return mcp_tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
            raise

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool on the MCP server.

        :param tool_name: Name of the tool to execute
        :param arguments: Arguments for the tool
        :return: Result from the tool execution
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            # Convert arguments to JSON string
            args_json = json.dumps(arguments)
            
            # Call the tool through MCP
            result = await self.session.call_tool(tool_name, args_json)
            return result.content
        except Exception as e:
            self.logger.error(f"Error executing MCP tool '{tool_name}': {e}")
            raise

    async def cleanup(self) -> None:
        """
        Clean up resources.
        """
        await self.exit_stack.aclose() 