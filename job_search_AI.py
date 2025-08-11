import ollama
import json
import re
from job_search import scrape_linkedin_jobs

QUERY_MODEL_NAME = "gemma2:2b"
RELEVANCE_MODEL_NAME = "qwen3:0.6b"
SUMMARY_MODEL_NAME = "qwen3:0.6b"

# --- User profile and keywords ---
user_profile = """
I want a PhD in reproduction. 
I'm a MSc graduate in Reproductive Biology and Physiology.
"""

keywords = [
    "Behavior",
    "behavioral reproduction",
    "evolution",
    "developmental biology",
    "Reproduction",
    "fertility",
    "breeding",
    "reproductive",
    "animal",
    "animal welfare",
    "embryology",
    "reproductive physiology",
    "conservation",
    "ecophysiology",
    "Cellular",
]

excluded_keywords = [
    "Molecular",
    "Postdoctoral",
    "internship",
    "senior",
    "workshop",
    "lab technician",
    "lecturer",
    "Computational",
    "Bioinformatics"
]

# --- Step 1: Generate search queries and detect location ---
def generate_queries(user_profile, keywords, excluded_keywords):
    query_prompt = f"""
You are an advanced opportunity search query generator for academic, research, and industry positions.

User profile:
{user_profile}

Keywords:
{", ".join(keywords)}

Excluded Keywords:
{", ".join(excluded_keywords)}

Your tasks:

1. **Extract the search location** from the profile.  
   - If no location is mentioned, return exactly "Worldwide".

2. **Understand the user's intent**:
   - Detect if the user explicitly requests a specific degree (e.g., PhD, Master's), job type (e.g., industry role, research), or conditions (e.g., remote, internship, fellowship).
   - If a stage/role is mentioned, prioritize those in the queries, but still include related opportunities at other career stages and contexts.

3. **Generate an extremely large and diverse list of queries** that could realistically appear in search engines, job boards, and academic listings.
   - **Do NOT include any query containing the excluded keywords**, even as part of longer phrases.
   - Include **ALL RELEVANT career/academic levels** that the user could want:
     - Degree programs: "PhD", "Doctorate", "Doctoral program", "Master’s", "MSc", "Bachelor’s", "BSc", "Undergraduate program"
     - Research roles: "Postdoctoral position", "Postdoc", "Research fellow", "Research scientist", "Principal investigator", "PI role", "Research coordinator", "Senior researcher", "Junior researcher"
     - Academic jobs: "Lecturer", "Assistant professor", "Associate professor", "Professor", "Tenure-track position", "Adjunct professor", "Visiting scholar"
     - Technical/assistant roles: "Research assistant", "Lab technician", "Field technician", "Data analyst", "Project manager"
     - Opportunities/funding: "Scholarship", "Fellowship", "Grant-funded research", "Traineeship", "Graduate assistantship"
     - Industry/other: "Industry research", "Consulting role", "Applied research", "Internship", "Entry-level", "Remote job", "Remote research"
     - Training/short-term: "Workshop", "Summer school", "Bootcamp", "Certification program"

4. **For EACH keyword**:
   - Generate role-specific queries (e.g., "PhD in {{keyword}}", "Postdoc in {{keyword}}", "Remote {{keyword}} research").
   - Generate **combined keyword queries**:
     - Pair or group related terms (e.g., "{{keyword1}} and {{keyword2}}", "Research in {{keyword1}} related to {{keyword2}}").
     - Include specializations (e.g., "computational {{keyword}}", "molecular {{keyword}}", "applied {{keyword}}", "environmental {{keyword}}").
   - Include field-specific variations and synonyms based on discipline.

5. **Ensure quality**:
   - Avoid duplicates (case-insensitive).
   - Keep results professional and realistic for actual job/program listings.
   - Use concise, search-friendly phrasing.

Output **only** valid JSON in this exact format (no explanations, no markdown):
{{
  "location": "<location or Worldwide>",
  "queries": [
    "<query1>",
    "<query2>",
    ...
  ]
}}
"""

    query_response = ollama.chat(
        model=QUERY_MODEL_NAME, messages=[{"role": "user", "content": query_prompt}]
    )
    return query_response["message"]["content"]


# queries_data = generate_queries(user_profile, keywords, excluded_keywords)


# --- Clean JSON if LLM wrapped it in code fences ---
def clean_queries(queries_data):
    queries_data_clean = re.search(r"\{.*\}", queries_data, re.DOTALL)
    if not queries_data_clean:
        raise ValueError("No valid JSON object found in LLM output:\n" + queries_data)
    return json.loads(queries_data_clean.group(0))


# data = clean_queries(queries_data)
# location = data["location"]
# queries = data["queries"]
# print("Generated Queries & Location:\n", json.dumps(data, indent=2))


# --- Run scraping in batches of 25 ---
def get_job_offers(queries, location):
    for i in range(0, 125, 25):
        scrape_linkedin_jobs(queries, location, i)


# get_job_offers(queries,location)


# --- Step 2: Job relevance checker ---
def is_job_relevant(job, user_profile, keywords, excluded_keywords):
    relevance_prompt = f"""
User profile:
{user_profile}

Included Keywords:
{", ".join(keywords)}

Excluded Keywords:
{", ".join(excluded_keywords)}

Job posting:
Title: {job['title']}
Summary: {job['summary']}

Instructions:
Determine if this opportunity is relevant to the user's career goals and interests, based on:
- Matching any of the user's keywords or closely related concepts.
- Matching the desired education path (even if the requirement is higher than the current degree, consider it relevant if the user *wants* that qualification).
- Matching field of work, research area, or subject matter.
- **Reject if the posting contains any excluded keyword or strongly implies it**.

Output only in this JSON format:
{{
    "relevant": true or false,
    "explanation": "one to two sentences ONLY"
}}
"""
    response = ollama.chat(
        model=RELEVANCE_MODEL_NAME,
        messages=[{"role": "user", "content": relevance_prompt}],
    )

    result_text = response["message"]["content"].strip()
    print("LLM Relevance Check Output:", result_text)

    try:
        result_json = json.loads(result_text)
        # Ensure both keys exist
        return {
            "relevant": result_json.get("relevant", False),
            "explanation": result_json.get("explanation", "").strip()
        }
    except json.JSONDecodeError:
        # Fallback if parsing fails
        if "false" in result_text.lower():
            return {"relevant": False, "explanation": "Model output parsing failed, marked as not relevant."}
        return {"relevant": True, "explanation": "Model output parsing failed, marked as relevant."}


def get_summary(job):
    summary_system = """
You are an assistant that analyzes job postings.
Summarize each posting into:
1. Key Requirements - list technical skills, degrees, certifications, and experience required.
2. Role Details - summarize main responsibilities, daily tasks, and objectives.
Keep it CONCISE and structured in bullet points.
"""
    job_text = f"""
Title: {job['title']}
Description: {job['description']}
Criteria: {job['criteria']}
"""
    response = ollama.chat(
        model=SUMMARY_MODEL_NAME,
        messages=[
            {"role": "system", "content": summary_system},
            {"role": "user", "content": job_text},
        ],
    )
    return response["message"]["content"].strip()


# --- Step 3: Load job data ---
with open("./offers/Jobs.txt", "r", encoding="utf-8") as f:
    job_info_lists = json.load(f)

# --- Step 4: Run relevance check ---
jobs_summary = []

for job_data in job_info_lists:
    summary = get_summary(job_data)
    print("summary: ", summary)

    relevance_result = is_job_relevant(
        {**job_data, "summary": summary}, user_profile, keywords, excluded_keywords
    )

    job_entry = {
        **job_data,
        "summary": summary,
        "relevant": relevance_result["relevant"],
        "explanation": relevance_result["explanation"]
    }
    jobs_summary.append(job_entry)

    # Write after each job so you can check progress
    with open("./offers/jobs_relevant.json", "w", encoding="utf-8") as f:
        json.dump(jobs_summary, f, ensure_ascii=False, indent=2)

    print("job done\n\n")




