from mcp import Tool, ServerSession
from mcp.server import Server
import json

# Define input/output schemas for our tools
math_input_schema = {
    "type": "object",
    "properties": {
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"}
    },
    "required": ["a", "b"]
}

math_output_schema = {
    "type": "object",
    "properties": {
        "result": {"type": "number", "description": "Result of the operation"}
    }
}

# Define math tool functions
async def add(a: float, b: float) -> dict:
    return {"result": a + b}

async def subtract(a: float, b: float) -> dict:
    return {"result": a - b}

async def multiply(a: float, b: float) -> dict:
    return {"result": a * b}

async def divide(a: float, b: float) -> dict:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return {"result": a / b}

# Create MCP server
server_instance = Server(name="math_server")

# Register tools
server_instance.register_tool(
    Tool(
        name="add",
        description="Add two numbers together",
        input_schema=math_input_schema,
        output_schema=math_output_schema,
        handler=add
    )
)

server_instance.register_tool(
    Tool(
        name="subtract",
        description="Subtract second number from first number",
        input_schema=math_input_schema,
        output_schema=math_output_schema,
        handler=subtract
    )
)

server_instance.register_tool(
    Tool(
        name="multiply",
        description="Multiply two numbers together",
        input_schema=math_input_schema,
        output_schema=math_output_schema,
        handler=multiply
    )
)

server_instance.register_tool(
    Tool(
        name="divide",
        description="Divide first number by second number",
        input_schema=math_input_schema,
        output_schema=math_output_schema,
        handler=divide
    )
)

if __name__ == "__main__":
    server_instance.run() 