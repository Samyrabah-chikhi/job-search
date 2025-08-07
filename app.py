import os
from dotenv import load_dotenv
# Class attributes
from typing import Annotated
from typing_extensions import TypedDict

# Graph builder 
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# Model 
from langchain_ollama import OllamaLLM

# Tools
from langchain_tavily import TavilySearch

# Model Init
llm = OllamaLLM(model="gemma3:1b",num_thread=4)

# Load keys
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") 

# Init tools
tool = TavilySearch(max_results=2)
tools = [tool]

# Graph builder
class State(TypedDict):
    messages: Annotated[list,add_messages]

graph_builder = StateGraph(State)

# add tools to LLM


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile()

# Showing the graph
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# Stream answers from LLM
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1])


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Assistant: Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
