import ollama
import json
import re
from job_search import scrape_linkedin_jobs
from pydantic import BaseModel, Field
from typing import List

# Model names
QUERY_MODEL_NAME = "gemma2:2b"
RELEVANCE_MODEL_NAME = "qwen3:0.6b"
SUMMARY_MODEL_NAME = "qwen3:0.6b"

# --- User profile and keywords ---
user_profile = """
I want a PhD in reproduction. 
I graduated with a Master's degree in Reproductive Biology and Physiology. 
My previous experience as a laboratory assistant has equipped me with skills in physico-chemical and microbiological analyses,
as well as good laboratory practices. 
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
    "Computational",
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

# Output exactly:
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
    for i in range(0, len(queries), 25):
        scrape_linkedin_jobs(queries[i : i + 25], location, i)


# --- Step 3: Relevance check ---
def is_job_relevant(
    job: dict, user_profile: str, keywords: List[str], excluded_keywords: List[str]
) -> RelevanceResult:

    relevance_prompt = f"""
User profile:
{user_profile}

Included Keywords (desired areas of focus):
{", ".join(keywords)}

Excluded Keywords (areas to avoid):
{", ".join(excluded_keywords)}

Job posting:
Title: {job['title']}
Summary: {job['summary']}

Instructions:
You are an expert in matching job postings to a user's career goals.

Evaluate this posting using BOTH the included and excluded keywords together:

1. **Included keywords** — Identify how many and how strongly they appear in the role description. 
   - Give more weight if they describe the main duties, research area, or goals.
   - Lower weight if they are only mentioned briefly or as optional skills.

2. **Excluded keywords** — Check if any are present.
   - If an excluded keyword is the main field or dominates the responsibilities → treat as a strong negative signal.
   - If it is secondary or minor, weigh it against the importance of included keywords.
   - If the job aligns strongly with the user's included keywords and ambitions despite minor excluded keyword mentions, it may still be relevant.

3. **Balancing decision**:
   - Accept if included keywords and the overall field match the user’s ambitions AND excluded keywords are minor/secondary.
   - Reject if excluded keywords dominate, even if some included keywords appear.
   - If the relevance is borderline, reduce confidence accordingly.

4. **User's profile**:
    - Take into account the user's needs and background.
    
Output strictly in JSON:
{{
    "relevant": true or false,
    "confidence": float between 0 and 1,
    "explanation": "one to two sentences ONLY explaining the reasoning, explicitly mentioning how both included and excluded keywords influenced the decision."
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
    # queries_data = generate_queries(user_profile, keywords, excluded_keywords)
    # location = queries_data.location
    # queries = queries_data.queries

    # 2. Scrape jobs
    # get_job_offers(queries, location)

    # 3. Load scraped jobs
    with open("./offers/Jobs.txt", "r", encoding="utf-8") as f:
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
                    "confidence: ": relevance_result.confidence,
                    "explanation": relevance_result.explanation,
                },
            }
        )
        # 5. Save all results
        with open("./offers/Job_relevant.json", "w", encoding="utf-8") as f:
            json.dump(jobs_summary, f, ensure_ascii=False, indent=2)

print("End.")
