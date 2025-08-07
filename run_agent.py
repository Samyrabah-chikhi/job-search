from agent_graph import graph

inital_state = {"messages": [{"role": "user", "content": "Search who invented AI"}]}

result = graph.invoke(inital_state)
print("\n📬 Agent Response:")
for msg in result["messages"]:
    print(f"{msg['role'].upper()}: {msg['content']}")
