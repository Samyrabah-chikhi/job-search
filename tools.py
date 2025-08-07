from ddgs import DDGS


def search(query: str) -> str:
    ddgs = DDGS()
    results = ddgs.text(query)
    try:
        return results
    except StopIteration:
        return "No results found."

