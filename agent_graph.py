from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from llm import chain
from tools import search


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def router_node(state: AgentState) -> str:
    user_msg = state["messages"][-1].content
    response = chain.invoke({"input": user_msg})
    print(f"🧠 LLM decided: {response}")

    if "search" in response.lower():
        return (state, "search")
    return (state, "search")


def search_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    result = search(query)
    print(f"🔍 Search result: {result}")

    return {"messages": state["messages"] + [{"role": "tool", "content": result}]}


builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("search", search_node)

builder.add_conditional_edges("router", router_node, {
    "search": "search"
})

builder.set_entry_point("router")
builder.add_edge("search", END)


graph = builder.compile()
