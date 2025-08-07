from ollama import ChatResponse, chat
from ddgs import DDGS

MODEL_NAME = "qwen3:0.6b"

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant, you're goal is to answer the user's question based on your available tools. if you can't answer the question with the tools answer with: I can't answer your question",
    }
]


def add_two_numbers(a: int, b: int):
    """function to add two integer numbers together
    args:
    a (int): the first number
    b (int): the second number

    return a + b"""

    return a + b


def search(query: str) -> str:
    """function to search online for a query
    args:
    query (string): the content you'll be searching
    
    output: result of the search (string)"""
    ddgs = DDGS()
    results = ddgs.text(query)
    try:
        return results
    except StopIteration:
        return "No results found."


available_tools = {"add_two_numbers": add_two_numbers, "search": search}
tools = [add_two_numbers,search]

while True:
    user_input = input("User: ")
    if user_input in ["/bye", "/quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    response: ChatResponse = chat(
        model=MODEL_NAME, messages=messages, tools=tools, think=True
    )

    print("Thinking:", response.message.thinking)
    print("Response:", response.message.content)

    if response.message.tool_calls:
        messages.append(response.message)

        for tool in response.message.tool_calls:
            func = available_tools.get(tool.function.name)
            if func:
                print("Calling function:", tool.function.name)
                print("Arguments:", tool.function.arguments)
                output = func(**tool.function.arguments)
                print("Output:", output)

                messages.append({
                    "role": "tool",
                    "content": str(output),
                    "tool_name": tool.function.name
                })

        response: ChatResponse = chat(
            model=MODEL_NAME, messages=messages, tools=tools, think=True
        )
        print("Thinking:", response.message.thinking)
        print("Final response:", response.message.content)
        messages.append(response.message)