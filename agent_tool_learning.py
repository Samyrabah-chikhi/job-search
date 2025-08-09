from ollama import ChatResponse, chat
from playwright.sync_api import sync_playwright

MODEL_NAME = "qwen3:1.7b"


# ----------------------
# Tool: LinkedIn Job Search
# ----------------------
def scrape_linkedin_jobs(query: str, location: str, limit: int = 10) -> list:
    limit = int(limit)
    url = f"https://www.linkedin.com/jobs/search/?keywords={query}&location={location}"
    jobs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        page.wait_for_selector(".jobs-search__results-list")

        job_list = page.query_selector(".jobs-search__results-list")

        for _ in range(5):
            page.evaluate("(el) => el.scrollBy(0, 1000)", job_list)
            page.wait_for_timeout(1000)

        listings = page.query_selector_all(".jobs-search__results-list li")[:limit]

        for li in listings:
            title = li.query_selector("h3").inner_text().strip()
            company = li.query_selector("h4").inner_text().strip()
            link = li.query_selector("a").get_attribute("href")
            jobs.append(
                {
                    "title": title,
                    "company": company,
                    "link": link,
                }
            )
        browser.close()

    return jobs


# ----------------------
# Available Tools
# ----------------------
AVAILABLE_TOOLS = {
    "scrape_linkedin_jobs": scrape_linkedin_jobs,
}

TOOLS = [scrape_linkedin_jobs]


# ----------------------
# Tool Execution
# ----------------------
def handle_tool_calls(tool_calls):
    outputs = []
    for tool in tool_calls:
        func = AVAILABLE_TOOLS.get(tool.function.name)
        if func:
            print(
                f"\n[Tool Name] {tool.function.name}\n[Arguments] {tool.function.arguments}"
            )
            output = func(**tool.function.arguments)
            print(f"[Tool Output] {output}")
            outputs.append(
                {
                    "role": "tool",
                    "content": str(output),
                    "tool_name": tool.function.name,
                }
            )
    return outputs


# ----------------------
# Recursive Streaming Chat
# ----------------------
def stream_until_done(messages, model=MODEL_NAME, tools=TOOLS, think=True):
    print("\nAssistant:")
    thinking = True
    response_stream = chat(
        model=model,
        messages=messages,
        tools=tools,
        think=thinking,
        stream=True,
    )

    full_response = ""
    tool_calls = []

    for chunk in response_stream:
        if chunk.message.thinking:
            if thinking:
                thinking = False
                print("<Think>\n")
            print(f"{chunk.message.thinking}", end="", flush=True)
        elif chunk.message.tool_calls:
            if not thinking:
                thinking = True
                print("</Think>\n")
            tool_calls.extend(chunk.message.tool_calls)
        elif chunk.message.content:
            if not thinking:
                thinking = True
                print("</Think>\n")
            print(chunk.message.content, end="", flush=True)
            full_response += chunk.message.content

    print()

    if tool_calls:
        outputs = handle_tool_calls(tool_calls)
        messages.extend(outputs)
        return stream_until_done(messages, model, tools, think)

    return full_response


# ----------------------
# Main Loop
# ----------------------
def main():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a job search assistant. Your ONLY purpose is to find job, internship, study "
                "or research offers on LinkedIn using the `scrape_linkedin_jobs` tool. "
                "For every request, you MUST have both a 'query' (job title, field, or role) "
                "and a 'location'. If the user does not provide one, ask for it or use Worldwide. "
                "If the request is not related to job searching, respond with: 'I can only help you search for jobs.'"
            ),
        }
    ]

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in {"/bye", "/quit", "/exit"}:
            break
        messages.append({"role": "user", "content": user_input})
        answer = stream_until_done(messages)
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
