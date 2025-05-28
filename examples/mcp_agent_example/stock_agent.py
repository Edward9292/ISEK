import asyncio
import os
from pathlib import Path
from isek.agent.single_agent import SingleAgent
from isek.agent.persona import Persona
from isek.agent.toolbox import ToolBox
from isek.llm.openai_model import OpenAIModel
from isek.agent.tools.mcp_handler import MCPHandler
from dotenv import load_dotenv

async def main():
    # Load environment variables
    load_dotenv()

    # Create a stock market expert persona
    persona_desc = {
        "name": "StockExpert",
        "bio": "A knowledgeable stock market expert",
        "lore": "I am an expert in stock market analysis and can provide real-time stock information and insights",
        "knowledge": "I have deep knowledge of stock markets, trading, and financial analysis",
        "routine": "I help users get information about stocks, analyze market trends, and compare different stocks"
    }
    persona = Persona.from_json(persona_desc)

    # Create LLM model
    model = OpenAIModel(
        model_name=os.environ.get("OPENAI_MODEL_NAME"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Create agent with the persona and model
    agent = SingleAgent(persona=persona, model=model)

    # Get the path to the MCP server script
    current_dir = Path(__file__).parent
    server_script = current_dir / "mcp_stock_server.py"

    # Initialize MCP handler and register tools
    if not agent.tool_manager.mcp_handler:
        agent.tool_manager.mcp_handler = MCPHandler()
    await agent.tool_manager._register_mcp_tools(str(server_script))

    # Example conversation
    messages = [
        "What's the current price of AAPL?",
        "Can you show me the stock history for MSFT over the last month?",
        "How does GOOGL compare to AMZN in terms of price?"
    ]

    for message in messages:
        print(f"\nUser: {message}")
        response = await agent.chat(message)
        print(f"Agent: {response}")

    # Cleanup
    await agent.tool_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 