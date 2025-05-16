import json
import inspect
import requests
from isek.agent.persona import Persona
from isek.util.logger import logger # Assuming logger has a standard logging interface
from typing import List, Dict, Callable, Any, Optional, Union


class ToolBox:
    """
    Manages a collection of tools (functions) that an agent can use.

    The ToolBox handles the registration of tools, storage of their metadata
    (like descriptions and schemas for LLM interaction), and the execution
    of tool calls based on requests from a language model.

    It supports both direct function tools and MCP-based tools, allowing for
    flexible and scalable tool management.
    """
    def __init__(self, persona: Optional[Persona] = None, mcp_server_url: Optional[str] = None) -> None:
        """
        Initializes a ToolBox instance.

        :param persona: The persona of the agent that will be using these tools.
                        Used for logging purposes. Defaults to None.
        :type persona: typing.Optional[isek.agent.persona.Persona]
        :param mcp_server_url: The base URL of the MCP server for dynamic tool discovery.
                              Defaults to None.
        :type mcp_server_url: typing.Optional[str]
        """
        self.logger = logger
        self.persona: Optional[Persona] = persona # Store persona for context in logging
        self.mcp_server_url = mcp_server_url.rstrip('/') if mcp_server_url else None

        # Tool containers
        self.all_tools: Dict[str, Callable[..., Any]] = {} # Maps tool name to callable function
        self.mcp_tools: Dict[str, Dict[str, Any]] = {} # Maps MCP tool name to its metadata

        # Tool metadata
        self.tool_descriptions: Dict[str, str] = {} # Maps tool name to its docstring
        self.tool_schemas: Dict[str, Dict[str, Any]] = {} # Maps tool name to its LLM-compatible schema

        # Initialize MCP tools if server URL is provided
        if self.mcp_server_url:
            self._discover_mcp_tools()

    def _log(self, message: str) -> None:
        """
        Logs a message using the configured logger, prepending persona name if available.

        :param message: The message to log.
        :type message: str
        """
        if self.logger:
            prefix = f"[{self.persona.name}] " if self.persona else ""
            self.logger.info(f"{prefix}ToolBox: {message}")

    def register_tool(self, func: Callable[..., Any]) -> None:
        """
        Registers a new callable function as a tool.

        The function's name is used as the tool name. Its docstring is
        used as the description, and an LLM-compatible schema is generated
        based on its signature and type annotations.

        :param func: The function to register as a tool. It should have type hints
                     for its parameters and a docstring for its description.
        :type func: typing.Callable[..., typing.Any]
        """
        name = func.__name__

        if name in self.all_tools:
            self._log(f"Warning: Tool '{name}' is being re-registered. Overwriting existing tool.")

        # Store the function
        self.all_tools[name] = func

        # Generate and store the schema
        try:
            self.tool_schemas[name] = self._function_to_schema(func)
        except (ValueError, KeyError) as e:
            self._log(f"Error generating schema for tool '{name}': {e}. Tool not fully registered.")
            # Optionally, decide if an incomplete registration is allowed or if it should raise
            if name in self.all_tools: del self.all_tools[name] # Rollback
            return

        # Store metadata (docstring)
        self.tool_descriptions[name] = (func.__doc__ or f"No description provided for tool {name}.").strip()

        self._log(f"Tool added: {name}")

    def register_tools(self, tools: List[Callable[..., Any]]) -> None:
        """
        Registers multiple tools at once.

        Iterates through the provided list of functions and calls
        :meth:`register_tool` for each one.

        :param tools: A list of callable functions to register as tools.
        :type tools: typing.List[typing.Callable[..., typing.Any]]
        """
        for tool in tools:
            self.register_tool(tool)

    def get_tool(self, name: str) -> Optional[Callable[..., Any]]:
        """
        Retrieves a registered tool function by its name.

        :param name: The name of the tool to retrieve.
        :type name: str
        :return: The callable tool function if found, otherwise `None`.
        :rtype: typing.Optional[typing.Callable[..., typing.Any]]
        """
        return self.all_tools.get(name)

    def get_tool_names(self) -> List[str]:
        """
        Gets a list of names of all registered tools.

        :return: A list of tool names.
        :rtype: typing.List[str]
        """
        return list(self.all_tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Gets the LLM-compatible schemas for all registered tools.

        .. note::
            The original method had an optional `category` parameter which is not
            used in the current implementation as tools are not categorized.
            This docstring reflects the current signature.

        :return: A list of tool schemas. Each schema is a dictionary.
        :rtype: typing.List[typing.Dict[str, typing.Any]]
        """
        return [self.tool_schemas[name] for name in self.all_tools.keys() if name in self.tool_schemas]

    def execute_tool_call(self, tool_call: Any, **extra_kwargs: Any) -> str:
        """
        Executes a tool call based on an object from an LLM response.

        The method first checks if the tool is a direct function tool, then falls back
        to checking if it's an MCP tool. If neither is found, returns an error message.

        :param tool_call: The tool call object, typically from an LLM's response.
        :type tool_call: typing.Any
        :param extra_kwargs: Additional keyword arguments to be passed to the tool function.
        :type extra_kwargs: typing.Any
        :return: A string representation of the tool's execution result.
        :rtype: str
        """
        try:
            name = tool_call.function.name
        except AttributeError:
            error_msg = "Invalid tool_call object: missing 'function.name' attribute."
            self._log(f"Error: {error_msg}")
            return error_msg

        # Try direct function tool first
        if name in self.all_tools:
            return self._execute_direct_tool(name, tool_call, extra_kwargs)
        
        # Try MCP tool if available
        if name in self.mcp_tools:
            return self._execute_mcp_tool(name, tool_call, extra_kwargs)
        
        error_msg = f"Tool '{name}' not found in either direct tools or MCP tools."
        self._log(f"Error: {error_msg}")
        return error_msg

    def _execute_direct_tool(self, name: str, tool_call: Any, extra_kwargs: Dict[str, Any]) -> str:
        """
        Executes a direct function tool.

        :param name: The name of the tool to execute.
        :type name: str
        :param tool_call: The tool call object.
        :type tool_call: typing.Any
        :param extra_kwargs: Additional keyword arguments.
        :type extra_kwargs: typing.Dict[str, typing.Any]
        :return: The result of the tool execution.
        :rtype: str
        """
        func = self.all_tools[name]
        try:
            arguments_json = tool_call.function.arguments
            if not isinstance(arguments_json, str):
                raise ValueError(f"Tool arguments for '{name}' must be a JSON string, got {type(arguments_json)}")

            args_from_llm = json.loads(arguments_json)
            if not isinstance(args_from_llm, dict):
                raise ValueError(f"Parsed tool arguments for '{name}' must be a dictionary, got {type(args_from_llm)}")

            final_args = {**args_from_llm, **extra_kwargs}
            self._log(f"Executing direct tool '{name}' with arguments: {final_args}")
            result = func(**final_args)
            return str(result) if result is not None else "Tool executed successfully with no return value."
        except Exception as e:
            error_msg = f"Error executing direct tool '{name}': {e}"
            self._log(f"Error: {error_msg}")
            return error_msg

    def _execute_mcp_tool(self, name: str, tool_call: Any, extra_kwargs: Dict[str, Any]) -> str:
        """
        Executes an MCP tool by making a request to the MCP server.

        :param name: The name of the tool to execute.
        :type name: str
        :param tool_call: The tool call object.
        :type tool_call: typing.Any
        :param extra_kwargs: Additional keyword arguments.
        :type extra_kwargs: typing.Dict[str, typing.Any]
        :return: The result of the MCP tool execution.
        :rtype: str
        """
        try:
            arguments_json = tool_call.function.arguments
            if not isinstance(arguments_json, str):
                raise ValueError(f"Tool arguments for '{name}' must be a JSON string, got {type(arguments_json)}")

            args_from_llm = json.loads(arguments_json)
            if not isinstance(args_from_llm, dict):
                raise ValueError(f"Parsed tool arguments for '{name}' must be a dictionary, got {type(args_from_llm)}")

            final_args = {**args_from_llm, **extra_kwargs}
            self._log(f"Executing MCP tool '{name}' with arguments: {final_args}")

            response = requests.post(
                f"{self.mcp_server_url}/execute/{name}",
                json=final_args
            )
            response.raise_for_status()
            result = response.json()
            return str(result)
        except Exception as e:
            error_msg = f"Error executing MCP tool '{name}': {e}"
            self._log(f"Error: {error_msg}")
            return error_msg

    def refresh_mcp_tools(self) -> None:
        """
        Refreshes the list of available MCP tools by querying the MCP server.
        """
        if self.mcp_server_url:
            self._discover_mcp_tools()
        else:
            self._log("Cannot refresh MCP tools: No MCP server URL configured")

    def _discover_mcp_tools(self) -> None:
        """
        Discovers available tools from the MCP server and caches their metadata.
        """
        try:
            response = requests.get(f"{self.mcp_server_url}/tools")
            response.raise_for_status()
            tools = response.json()
            
            for tool in tools:
                name = tool.get('name')
                if name:
                    self.mcp_tools[name] = tool
                    self.tool_schemas[name] = self._mcp_tool_to_schema(tool)
                    self.tool_descriptions[name] = tool.get('description', f'No description for MCP tool {name}')
            
            self._log(f"Discovered {len(self.mcp_tools)} tools from MCP server")
        except Exception as e:
            self._log(f"Error discovering MCP tools: {e}")

    def _mcp_tool_to_schema(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts an MCP tool definition into an LLM-compatible schema.

        :param tool: The MCP tool definition.
        :type tool: typing.Dict[str, typing.Any]
        :return: The LLM-compatible schema for the tool.
        :rtype: typing.Dict[str, typing.Any]
        """
        return {
            "type": "function",
            "function": {
                "name": tool['name'],
                "description": tool.get('description', f'No description for MCP tool {tool["name"]}'),
                "parameters": {
                    "type": "object",
                    "properties": tool.get('parameters', {}),
                    "required": tool.get('required', []),
                },
            },
        }

    def _function_to_schema(self, func: Callable[..., Any]) -> Dict[str, Any]:
        """
        Converts a Python function into an LLM-compatible tool schema.

        This schema typically follows a format similar to OpenAI's function calling
        schema, detailing the function's name, description (from its docstring),
        and parameters (derived from its signature and type annotations).

        Supported Python types for parameters are mapped to JSON schema types:
        `str` -> "string", `int` -> "integer", `float` -> "number",
        `bool` -> "boolean", `list` -> "array", `dict` -> "object",
        `NoneType` -> "null". Unannotated parameters or parameters with
        unsupported annotations default to "string".

        :param func: The callable function to convert.
        :type func: typing.Callable[..., typing.Any]
        :return: A dictionary representing the tool schema.
        :rtype: typing.Dict[str, typing.Any]
        :raises ValueError: If the function signature cannot be inspected or
                            if a parameter's type annotation is of a type that
                            cannot be directly mapped (and is not a common built-in).
        :raises KeyError: If an internal error occurs mapping type annotations. (Less likely with defaults)
        """
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array", # Note: This doesn't specify item types for the array.
            dict: "object", # Note: This doesn't specify properties for the object.
            type(None): "null",
            # For Union types, one might pick the first non-NoneType, or handle more complexly.
            # For Optional[X] (which is Union[X, NoneType]), this basic map won't inherently make it optional.
            # The 'required' list handles whether a parameter is optional from a calling perspective.
        }

        try:
            signature = inspect.signature(func)
        except ValueError as e: # e.g., for built-in functions in C
            raise ValueError(
                f"Failed to get signature for function '{func.__name__}': {e}"
            )

        parameters_properties: Dict[str, Dict[str, str]] = {}
        for param in signature.parameters.values():
            if param.name == 'self' and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD: # Skip self for methods
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL or param.kind == inspect.Parameter.VAR_KEYWORD: # Skip *args, **kwargs
                continue

            param_type_annotation = param.annotation
            json_type = "string" # Default type

            # Handle Optional[T] and Union[T, None]
            if hasattr(param_type_annotation, '__origin__') and param_type_annotation.__origin__ is Union:
                # Filter out NoneType for Optional fields
                args = [arg for arg in param_type_annotation.__args__ if arg is not type(None)]
                if len(args) == 1: # This was Optional[X] or Union[X, None]
                    param_type_annotation = args[0]
                # else: complex Union, default to string or handle as needed

            if param_type_annotation is not inspect.Parameter.empty:
                json_type = type_map.get(param_type_annotation, "string") # Default to string if type not in map
            
            # Basic description from annotation if possible, could be expanded
            param_description = f"Parameter '{param.name}' of type {param_type_annotation}"
            if param.default != inspect.Parameter.empty:
                param_description += f" (default: {param.default})"

            parameters_properties[param.name] = {"type": json_type, "description": param_description}

        required = [
            param.name
            for param in signature.parameters.values()
            if param.default == inspect.Parameter.empty and
               param.name != 'self' and # Ensure self is not in required
               param.kind != inspect.Parameter.VAR_POSITIONAL and # *args not required
               param.kind != inspect.Parameter.VAR_KEYWORD # **kwargs not required
        ]

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": (func.__doc__ or f"No description provided for tool {func.__name__}.").strip(),
                "parameters": {
                    "type": "object",
                    "properties": parameters_properties,
                    "required": required,
                },
            },
        }
    

class Tool:
    """Abstract base class for tools that can be used by agents."""
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema describing this tool."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        pass


class FunctionTool(Tool):
    """Tool implementation that wraps a callable function."""

    def __init__(self, func: Callable):
        """Initialize with a callable function.
        
        Args:
            func: The function to wrap as a tool
        """
        self.func = func
        self._schema = self._build_schema()

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for the wrapped function."""
        return self._schema

    def execute(self, *args, **kwargs) -> Any:
        """Execute the wrapped function."""
        return self.func(*args, **kwargs)

    def _build_schema(self) -> Dict[str, Any]:
        """Build the JSON schema for the wrapped function."""
        signature = inspect.signature(self.func)
        # Reuse the schema building logic from above
        parameters_properties: Dict[str, Dict[str, str]] = {}
        for param in signature.parameters.values():
            if param.name == 'self' and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL or param.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            param_type_annotation = param.annotation
            json_type = "string"

            if hasattr(param_type_annotation, '__origin__') and param_type_annotation.__origin__ is Union:
                args = [arg for arg in param_type_annotation.__args__ if arg is not type(None)]
                if len(args) == 1:
                    param_type_annotation = args[0]

            if param_type_annotation is not inspect.Parameter.empty:
                json_type = type_map.get(param_type_annotation, "string")
            
            param_description = f"Parameter '{param.name}' of type {param_type_annotation}"
            if param.default != inspect.Parameter.empty:
                param_description += f" (default: {param.default})"

            parameters_properties[param.name] = {"type": json_type, "description": param_description}

        required = [
            param.name
            for param in signature.parameters.values()
            if param.default == inspect.Parameter.empty and
               param.name != 'self' and
               param.kind != inspect.Parameter.VAR_POSITIONAL and
               param.kind != inspect.Parameter.VAR_KEYWORD
        ]

        return {
            "type": "function",
            "function": {
                "name": self.func.__name__,
                "description": (self.func.__doc__ or f"No description provided for tool {self.func.__name__}.").strip(),
                "parameters": {
                    "type": "object",
                    "properties": parameters_properties,
                    "required": required,
                },
            },
        }


class MCPTool(Tool):
    """Tool implementation that wraps an MCP server endpoint."""

    def __init__(self, server_url: str, endpoint: str, method: str = "POST"):
        """Initialize with MCP server details.
        
        Args:
            server_url: Base URL of the MCP server
            endpoint: The endpoint to call
            method: HTTP method to use (default: POST)
        """
        self.server_url = server_url.rstrip('/')
        self.endpoint = endpoint.lstrip('/')
        self.method = method
        self._schema = self._build_schema()

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for the MCP endpoint."""
        return self._schema

    def execute(self, *args, **kwargs) -> Any:
        """Execute the MCP endpoint call."""
        url = f"{self.server_url}/{self.endpoint}"
        response = requests.request(self.method, url, json=kwargs)
        response.raise_for_status()
        return response.json()

    def _build_schema(self) -> Dict[str, Any]:
        """Build the JSON schema for the MCP endpoint."""
        # This could be enhanced to fetch schema from MCP server if available
        return {
            "type": "function",
            "function": {
                "name": self.endpoint,
                "description": f"MCP endpoint {self.method} {self.endpoint}",
                "parameters": {
                    "type": "object",
                    "properties": {},  # Could be populated from MCP server schema
                    "required": [],
                },
            },
        }