import json
import inspect
import asyncio
from isek.agent.persona import Persona
from isek.util.logger import logger
from typing import List, Dict, Callable, Any, Optional, Union
from isek.agent.tools.base import BaseTool, FunctionTool
from isek.agent.tools.mcp_handler import MCPHandler

class ToolBox:
    """
    Manages a collection of tools that an agent can use.
    Provides a unified interface for registering and executing both local function tools
    and MCP server tools.
    """
    def __init__(self, persona: Optional[Persona] = None) -> None:
        """
        Initializes a ToolBox instance.

        :param persona: The persona of the agent that will be using these tools.
                        Used for logging purposes. Defaults to None.
        :type persona: typing.Optional[isek.agent.persona.Persona]
        """
        self.logger = logger
        self.persona: Optional[Persona] = persona

        # Tool containers
        self.all_tools: Dict[str, BaseTool] = {}
        
        # MCP handler
        self.mcp_handler: Optional[MCPHandler] = None

    async def register_tools(self, tools: Union[List[Union[Callable[..., Any], BaseTool]], str]) -> None:
        """
        Registers multiple tools at once. Can handle both local function tools and MCP server tools.

        :param tools: Either:
            - A list of callable functions or BaseTool instances for local tools
            - A string path to an MCP server script (.py or .js) for MCP tools
        :type tools: Union[List[Union[Callable[..., Any], BaseTool]], str]
        """
        if isinstance(tools, str):
            # Handle MCP server tools
            await self._register_mcp_tools(tools)
        else:
            # Handle local function tools
            for tool in tools:
                self.register_tool(tool)

    async def _register_mcp_tools(self, server_script_path: str) -> None:
        """
        Register tools from an MCP server.

        :param server_script_path: Path to the server script (.py or .js)
        """
        try:
            # Initialize MCP handler if not already initialized
            if not self.mcp_handler:
                self.mcp_handler = MCPHandler()

            # Connect to server and get tools
            await self.mcp_handler.connect_to_server(server_script_path)
            mcp_tools = await self.mcp_handler.get_available_tools()

            # Register all MCP tools
            for tool in mcp_tools:
                self.register_tool(tool)
            
            self.logger.info(f"Registered MCP tools: {[tool.name for tool in mcp_tools]}")
        except Exception as e:
            self.logger.error(f"Failed to register MCP tools: {e}")
            raise

    def _log(self, message: str) -> None:
        """
        Logs a message using the configured logger, prepending persona name if available.

        :param message: The message to log.
        :type message: str
        """
        if self.logger:
            prefix = f"[{self.persona.name}] " if self.persona else ""
            self.logger.info(f"{prefix}ToolBox: {message}")

    def register_tool(self, tool: Union[Callable[..., Any], BaseTool]) -> None:
        """
        Registers a new tool.

        :param tool: Either a callable function or a BaseTool instance
        :type tool: Union[Callable[..., Any], BaseTool]
        """
        if isinstance(tool, BaseTool):
            name = tool.name
            tool_instance = tool
        else:
            name = tool.__name__
            tool_instance = FunctionTool(tool)

        if name in self.all_tools:
            self._log(f"Warning: Tool '{name}' is being re-registered. Overwriting existing tool.")

        self.all_tools[name] = tool_instance
        self._log(f"Tool added: {name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Retrieves a registered tool by its name.

        :param name: The name of the tool to retrieve.
        :type name: str
        :return: The tool if found, otherwise None.
        :rtype: Optional[BaseTool]
        """
        return self.all_tools.get(name)

    def get_tool_names(self) -> List[str]:
        """
        Gets a list of names of all registered tools.

        :return: A list of tool names.
        :rtype: List[str]
        """
        return list(self.all_tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Gets the LLM-compatible schemas for all registered tools.

        :return: A list of tool schemas.
        :rtype: List[Dict[str, Any]]
        """
        return [tool.to_schema() for tool in self.all_tools.values()]

    async def execute_tool_call_async(self, tool_call: Any, **extra_kwargs: Any) -> str:
        """
        Executes a tool call based on an object from an LLM response asynchronously.

        :param tool_call: The tool call object, typically from an LLM's response
        :type tool_call: Any
        :param extra_kwargs: Additional keyword arguments to be passed to the tool
        :type extra_kwargs: Any
        :return: A string representation of the tool's execution result
        :rtype: str
        """
        try:
            name = tool_call.function.name
        except AttributeError:
            error_msg = "Invalid tool_call object: missing 'function.name' attribute."
            self._log(f"Error: {error_msg}")
            return error_msg

        tool = self.get_tool(name)
        if not tool:
            error_msg = f"Tool '{name}' not found."
            self._log(f"Error: {error_msg}")
            return error_msg

        try:
            # Ensure arguments is a string before trying to load JSON
            arguments_json = tool_call.function.arguments
            if not isinstance(arguments_json, str):
                raise ValueError(f"Tool arguments for '{name}' must be a JSON string, got {type(arguments_json)}")

            args_from_llm = json.loads(arguments_json)
            if not isinstance(args_from_llm, dict):
                raise ValueError(f"Parsed tool arguments for '{name}' must be a dictionary, got {type(args_from_llm)}")

            # Merge LLM args with any extra_kwargs, extra_kwargs take precedence
            final_args = {**args_from_llm, **extra_kwargs}

            self._log(f"Executing tool '{name}' with arguments: {final_args}")
            
            # Execute tool asynchronously if it supports it
            if hasattr(tool, 'execute_async'):
                result = await tool.execute_async(**final_args)
            else:
                result = tool.execute(**final_args)
                
            # Ensure result is stringifiable for consistent return type
            return str(result) if result is not None else "Tool executed successfully with no return value."
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON arguments for tool '{name}': {e}. Arguments: '{tool_call.function.arguments}'"
            self._log(f"Error: {error_msg}")
            return error_msg
        except TypeError as e:
            error_msg = f"Type error executing tool '{name}': {e}. Check tool signature and provided arguments."
            self._log(f"Error: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error executing tool '{name}': {e}"
            self._log(f"Error: {error_msg}")
            return error_msg

    def execute_tool_call(self, tool_call: Any, **extra_kwargs: Any) -> str:
        """
        Executes a tool call based on an object from an LLM response.
        This is a synchronous wrapper around the async execution.

        :param tool_call: The tool call object, typically from an LLM's response
        :type tool_call: Any
        :param extra_kwargs: Additional keyword arguments to be passed to the tool
        :type extra_kwargs: Any
        :return: A string representation of the tool's execution result
        :rtype: str
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute_tool_call_async(tool_call, **extra_kwargs))

    async def cleanup(self) -> None:
        """
        Clean up resources.
        """
        if self.mcp_handler:
            await self.mcp_handler.cleanup()