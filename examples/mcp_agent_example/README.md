# MCP Agent Example

This example demonstrates how to use MCP (Mission Control Protocol) servers as external tools with an ISEK agent. The example includes:

1. A simple MCP server (`mcp_math_server.py`) that provides basic math operations
2. An agent (`math_agent.py`) that uses these MCP tools to perform calculations

## Prerequisites

- Python 3.8+
- ISEK package installed
- MCP package installed (`pip install mcp`)

## Files

- `mcp_math_server.py`: An MCP server that provides math operations (add, subtract, multiply, divide)
- `math_agent.py`: An agent that uses the MCP math server as external tools

## Running the Example

1. Make sure you have all prerequisites installed
2. Run the agent:
   ```bash
   python math_agent.py
   ```

The agent will:
1. Start up and connect to the MCP math server
2. Register the available math tools
3. Run through a series of example calculations
4. Clean up resources when done

## Example Output

You should see something like:

```
User: What is 15 plus 7?
Agent: Let me calculate that for you. 15 + 7 = 22

User: Now multiply that result by 3
Agent: I'll multiply 22 by 3. 22 * 3 = 66

User: Finally, divide the result by 2
Agent: I'll divide 66 by 2. 66 / 2 = 33
```

## How It Works

1. The MCP server (`mcp_math_server.py`) defines four math operations as tools:
   - `add`: Adds two numbers
   - `subtract`: Subtracts second number from first
   - `multiply`: Multiplies two numbers
   - `divide`: Divides first number by second

2. The agent (`math_agent.py`):
   - Creates a math-focused persona
   - Connects to the MCP server
   - Registers the available math tools
   - Uses these tools to perform calculations based on user input

This example demonstrates how to:
- Create an MCP server with custom tools
- Connect an agent to an MCP server
- Use MCP tools in agent conversations
- Handle cleanup of MCP resources 