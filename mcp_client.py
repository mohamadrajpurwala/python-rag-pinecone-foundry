from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import os
import asyncio


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
)

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL"
)


# ============================================================
# Azure OpenAI LLM
# ============================================================

llm = ChatOpenAI(
    model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1",
    api_key=AZURE_OPENAI_API_KEY
)


# ============================================================
# MCP + LLM
# ============================================================

async def create_agent():

    # --------------------------------------------------------
    # Connect to MCP Server
    # --------------------------------------------------------

    mcp_client = MultiServerMCPClient(
        {
            "ticketing": {
                "transport": "streamable_http",
                "url": MCP_SERVER_URL
            }
        }
    )

    # --------------------------------------------------------
    # Get tools from MCP server
    # --------------------------------------------------------

    tools = await mcp_client.get_tools()

    print("\nAvailable MCP Tools:")

    for tool in tools:
        print(f"  - {tool.name}")

    # --------------------------------------------------------
    # Create Agent
    # --------------------------------------------------------

    agent = create_react_agent(
        llm,
        tools
    )

    return agent


# ============================================================
# Ask question
# ============================================================

async def ask_question(agent, question):

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    print("\n----------------------------------------")
    print("Question:")
    print(question)

    print("\nAnswer:")
    print(response["messages"][-1].content)

    print("----------------------------------------")


# ============================================================
# Main
# ============================================================

async def main():

    agent = await create_agent()

    while True:

        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            break

        await ask_question(
            agent,
            question
        )


if __name__ == "__main__":
    asyncio.run(main())