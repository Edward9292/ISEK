from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
import json
import asyncio

class BaseTool(ABC):
    """
    Abstract base class for all tools in the system.
    This can be implemented by both function-based tools and MCP-based tools.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with the given arguments.
        
        :param kwargs: Arguments for the tool
        :return: Result of the tool execution
        """
        pass

    def to_schema(self) -> Dict[str, Any]:
        """
        Convert the tool to an LLM-compatible schema.
        
        :return: Dictionary containing the tool's schema
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self._get_parameter_properties(),
                    "required": self._get_required_parameters()
                }
            }
        }

    @abstractmethod
    def _get_parameter_properties(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the properties of the tool's parameters.
        
        :return: Dictionary mapping parameter names to their properties
        """
        pass

    @abstractmethod
    def _get_required_parameters(self) -> list[str]:
        """
        Get the list of required parameters.
        
        :return: List of required parameter names
        """
        pass

class FunctionTool(BaseTool):
    """
    Tool implementation that wraps a Python function.
    """
    def __init__(self, func: Callable[..., Any], name: Optional[str] = None, description: Optional[str] = None):
        """
        Initialize a function-based tool.
        
        :param func: The function to wrap
        :param name: Optional name override (defaults to function name)
        :param description: Optional description override (defaults to function docstring)
        """
        super().__init__(
            name=name or func.__name__,
            description=description or (func.__doc__ or f"No description provided for {func.__name__}")
        )
        self.func = func
        self._signature = inspect.signature(func)

    def execute(self, **kwargs: Any) -> Any:
        """
        Execute the wrapped function with the given arguments.
        
        :param kwargs: Arguments for the function
        :return: Result of the function execution
        """
        return self.func(**kwargs)

    def _get_parameter_properties(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the properties of the function's parameters.
        
        :return: Dictionary mapping parameter names to their properties
        """
        properties = {}
        for name, param in self._signature.parameters.items():
            if name == 'self':
                continue
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            properties[name] = {
                "type": self._python_type_to_json_type(param_type),
                "description": f"Parameter '{name}' of type {param_type}"
            }
        return properties

    def _get_required_parameters(self) -> list[str]:
        """
        Get the list of required parameters.
        
        :return: List of required parameter names
        """
        return [
            name for name, param in self._signature.parameters.items()
            if name != 'self' and param.default == inspect.Parameter.empty
        ]

    def _python_type_to_json_type(self, python_type: type) -> str:
        """
        Convert a Python type to a JSON schema type.
        
        :param python_type: The Python type to convert
        :return: JSON schema type string
        """
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null"
        }
        return type_map.get(python_type, "string")

class MCPTool(BaseTool):
    """
    Tool implementation that communicates with an MCP server.
    """
    def __init__(self, 
                 name: str, 
                 description: str,
                 mcp_client: Any,  # Type will be defined when MCP client is implemented
                 parameter_properties: Dict[str, Dict[str, Any]],
                 required_parameters: list[str]):
        """
        Initialize an MCP-based tool.
        
        :param name: Name of the tool
        :param description: Description of the tool
        :param mcp_client: MCP client instance
        :param parameter_properties: Properties of the tool's parameters
        :param required_parameters: List of required parameter names
        """
        super().__init__(name=name, description=description)
        self.mcp_client = mcp_client
        self._parameter_properties = parameter_properties
        self._required_parameters = required_parameters

    async def execute_async(self, **kwargs: Any) -> Any:
        """
        Execute the tool by sending a request to the MCP server asynchronously.
        
        :param kwargs: Arguments for the tool
        :return: Result from the MCP server
        """
        return await self.mcp_client.execute_tool(self.name, kwargs)

    def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool by sending a request to the MCP server.
        This is a synchronous wrapper around the async execution.
        
        :param kwargs: Arguments for the tool
        :return: Result from the MCP server
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute_async(**kwargs))

    def _get_parameter_properties(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the properties of the tool's parameters.
        
        :return: Dictionary mapping parameter names to their properties
        """
        return self._parameter_properties

    def _get_required_parameters(self) -> list[str]:
        """
        Get the list of required parameters.
        
        :return: List of required parameter names
        """
        return self._required_parameters 