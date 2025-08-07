from browser_use.llm import ChatOllama
from browser_use import Agent
import ollama
import asyncio

llm = ChatOllama(model="gemma3")

async def main():
    agent = Agent(
        task="Navigate to https://www.google.com, search for GPT-4o price, then search for DeepSeek-V3 price, and compare them.",
        llm=llm,
        timeout=180
    )
    result = await agent.run()
    print(result)

asyncio.run(main())