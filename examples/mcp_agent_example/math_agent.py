import asyncio
import os
from pathlib import Path
from isek.agent.single_agent import SingleAgent
from isek.agent.persona import Persona
from isek.agent.toolbox import ToolBox

async def main():
    # Create a math-focused persona
    persona = Persona(
        name="MathWizard",
        description="A helpful agent that can perform mathematical calculations using MCP tools"
    )

    # Create agent with the persona
    agent = SingleAgent(persona=persona)

    # Get the path to the MCP server script
    current_dir = Path(__file__).parent
    server_script = current_dir / "mcp_math_server.py"

    # Register the MCP tools
    await agent.toolbox.register_tools(str(server_script))

    # Example conversation
    messages = [
        "What is 15 plus 7?",
        "Now multiply that result by 3",
        "Finally, divide the result by 2"
    ]

    for message in messages:
        print(f"\nUser: {message}")
        response = await agent.chat(message)
        print(f"Agent: {response}")

    # Cleanup
    await agent.toolbox.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 