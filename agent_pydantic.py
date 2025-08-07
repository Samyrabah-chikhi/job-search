import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.tools import Tool

@Tool
def search(query: str) -> list[str]:
    url = f"https://duckduckgo.com/?q={query}&format=xml"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'xml')
    results = []
    for result in soup.find_all('result'):
        title = result.find('title').text
        url = result.find('url').text
        results.append(f"{title}: {url}")
    return results
class CapitalResult(BaseModel):
    capital: str
agent = Agent(
    'ollama:qwen3:0.6b',
    result_type=CapitalResult,
    tools=[search],
    system_prompt="You are an assistant that answers questions by searching the web and extracting relevant information from search results."
)
result = agent.run_sync('What is the capital of France?')
print(result.data)