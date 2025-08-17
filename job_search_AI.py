import ollama
import json
from job_search import scrape_linkedin_jobs
from pydantic import BaseModel
from typing import List

# Constants
MAX_SCRAPE = 125

# Model names
QUERY_MODEL_NAME = "qwen3:0.6b"
RELEVANCE_MODEL_NAME = "qwen3:0.6b"
SUMMARY_MODEL_NAME = "qwen3:0.6b"

# --- User profile and keywords ---
user_profile = """
I want a PhD in reproduction. 
I graduated with a Master's degree in Reproductive Biology and Physiology.  
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
    "Plant",
    "Postdoctoral",
    "internship",
    "senior",
    "workshop",
    "lab technician",
    "lecturer",
    "Computational Biology",
    "Bioinformatics",
]


# ---------------- Models for structured output ----------------
class QueryOutput(BaseModel):
    location: str
    queries: List[str]


class RelevanceResult(BaseModel):
    relevant: bool
    explanation: str
    confidence: float


class JobSummary(BaseModel):
    key_requirements: List[str]
    role_details: List[str]


# --- Step 1: Generate search queries and detect location ---
def generate_queries(
    user_profile: str, keywords: List[str], excluded_keywords: List[str]
) -> QueryOutput:

    query_prompt = f"""
You create short, realistic search queries for job or academic opportunities.

User profile:
{user_profile}

INCLUDE keywords:
{", ".join(keywords)}

EXCLUDE keywords:
{", ".join(excluded_keywords)}

Rules:
0. Generate an extremely large and diverse list of queries** that could realistically appear in search engines, job boards, and academic listings.
1. Only generate queries relevant to the user profile and included keywords.
2. Never include any excluded keyword in any form, Even if part of a longer phrase.
3. Match the user's career stage. If they request a specific stage (e.g., PhD, Master's, entry-level, junior), only generate queries for that stage or equivalent — do not include higher or lower stages.
4. Combine included keywords naturally (e.g., "PhD in X", "Research on X and Y", "Internship in X").
5. Short queries:** 2-5 words max (e.g., "PhD [keyword]", "Remote [keyword]") so that more results appear in the search.
6. Use concise, search-friendly phrasing.
7. Keep results professional and realistic for actual job/program listings.
8. No duplicates.

Output JSON only:
{{
  "location": "<location or Worldwide>",
  "queries": ["<query1>", "<query2>", ...]
}}
"""

    query_response = ollama.chat(
        model=QUERY_MODEL_NAME,
        messages=[{"role": "user", "content": query_prompt}],
        format=QueryOutput.model_json_schema(),
    )

    queries_data = QueryOutput.model_validate_json(query_response.message.content)
    return queries_data


# --- Step 2: Scrape jobs ---
def get_job_offers(queries: List[str], location: str):
    for i in range(0, MAX_SCRAPE, 25):
        scrape_linkedin_jobs(queries, location, i)


# --- Step 3: Relevance check ---
def is_job_relevant(
    job: dict, user_profile: str, keywords: List[str], excluded_keywords: List[str]
) -> RelevanceResult:

    relevance_prompt = f"""
User profile:
{user_profile}

Included Keywords: {", ".join(keywords)}
Excluded Keywords: {", ".join(excluded_keywords)}

Job posting:
Title: {job['title']}
Summary: {job['summary']}

Decide if this job matches the user's career field and level.

Rules:
- Primary check: Does the role belong to the same career field as the user’s goals? If not, irrelevant.
- Career level: Exact match = best; higher/lower than desired = lower confidence; far outside = irrelevant.
- Use included keywords to confirm relevance within the career; weight central mentions higher.
- Excluded keywords: if central to the role → reject; if minor, lower confidence.
- Never mark as relevant if the field or goals don't align, even if some keywords match.

Output JSON:
{{
    "relevant": true/false,
    "confidence": float 0-1,
    "explanation": "1-2 sentences explaining how career field, level, and keywords influenced the decision."
}}
"""

    response = ollama.chat(
        model=RELEVANCE_MODEL_NAME,
        messages=[{"role": "user", "content": relevance_prompt}],
        format=RelevanceResult.model_json_schema(),
    )
    relevance = RelevanceResult.model_validate_json(response.message.content)
    return relevance


# --- Step 4: Summarizer ---
def get_summary(job: dict) -> JobSummary:
    summary_system = """
You are an assistant that analyzes job postings.
Summarize each posting into:
1. Key Requirements - technical skills, degrees, certifications, and experience required.
2. Role Details - main responsibilities, daily tasks, and objectives.
Keep it concise in bullet points.
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
        format=JobSummary.model_json_schema(),
    )
    return JobSummary.model_validate_json(response.message.content)


# ------------------- Main pipeline -------------------
if __name__ == "__main__":
    # 1. Generate queries
    queries_data = generate_queries(user_profile, keywords, excluded_keywords)
    location = queries_data.location
    queries = queries_data.queries
    
    print("Queries: ",queries)
    print("Location: ", location)
    # 2. Scrape jobs
    get_job_offers(queries, location)

    # 3. Load scraped jobs
    with open("./offers/Jobs_.txt", "r", encoding="utf-8") as f:
        job_info_lists = json.load(f)

    jobs_summary = []
    # 4. Process each job
    for job_data in job_info_lists:
        summary = get_summary(job_data)
        print("\nkey_requirements", summary.key_requirements, "\n")
        print("role_details", summary.role_details, "\n")

        relevance_result = is_job_relevant(
            {**job_data, "summary": summary}, user_profile, keywords, excluded_keywords
        )
        print("relevant: ", relevance_result.relevant, "\n")
        print("confidence: ", relevance_result.confidence, "\n")
        print("explanation: ", relevance_result.explanation, "\n")

        jobs_summary.append(
            {
                **job_data,
                "summary": {
                    "key_requirements": summary.key_requirements,
                    "role_details": summary.role_details,
                },
                "relevance": {
                    "relevant": relevance_result.relevant,
                    "confidence": relevance_result.confidence,
                    "explanation": relevance_result.explanation,
                },
            }
        )
        # 5. Save all results
        with open("./offers/Jobs_relevant_.json", "w", encoding="utf-8") as f:
            json.dump(jobs_summary, f, ensure_ascii=False, indent=2)

print("End.")
