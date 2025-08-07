from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

llm = OllamaLLM(model="gemma3:1b")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a smart agent. When you see a question that needs information, respond with 'search'.",
        ),
        ("user", "{input}"),
    ]
)

chain: Runnable = prompt | llm
