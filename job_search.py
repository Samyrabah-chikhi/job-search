import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import quote
import os

def scrape_linkedin_jobs(queries, location="Worldwide", start_index=0):
    """
    Scrape LinkedIn job postings for multiple queries.

    Args:
        queries (list): A list of search queries.
        location (str): Location to search in (default "Worldwide").
        start_index (int): Starting index for pagination (default 0).

    Returns:
        list: A list of job dictionaries with extracted details.
    """
    # Load existing jobs if file exists
    if os.path.exists("./offers/Jobs_.txt"):
        with open("./offers/Jobs_.txt", "r", encoding="utf-8") as f:
            try:
                job_info_lists = json.load(f)
            except json.JSONDecodeError:
                job_info_lists = []
    else:
        job_info_lists = []

    for query in queries:
        search_term = quote(query)
        job_list_url = (
            f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
            f"?keywords={search_term}&location={quote(location)}"
            f"&trk=public_jobs_jobs-search-bar_search-submit&pageNum=0&start={start_index}"
        )

        response = requests.get(job_list_url)
        if response.status_code != 200:
            print(f"⚠ Failed to fetch jobs for '{query}', status: {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        page_jobs = soup.find_all("li")

        id_lists = []
        for job in page_jobs:
            div = job.find("div", class_="base-card")
            if not div:
                continue
            job_id = div.get("data-entity-urn").split(":")[-1]
            id_lists.append(job_id)

        for job_id in id_lists:
            job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
            job_response = requests.get(job_url)
            if job_response.status_code != 200:
                continue

            job_soup = BeautifulSoup(job_response.text, 'html.parser')

            def safe_text(tag):
                return tag.text.strip() if tag else "Unknown"

            title = safe_text(job_soup.find("h2", class_="top-card-layout__title"))
            description = safe_text(job_soup.find("div", class_="show-more-less-html__markup"))
            criteria = safe_text(job_soup.find("ul", class_="description__job-criteria-list"))

            organisation_tag = job_soup.find("a", class_="topcard__org-name-link")
            organisation_name = safe_text(organisation_tag)
            organisation_url = organisation_tag.get("href") if organisation_tag else "Unknown"

            posted_time = safe_text(job_soup.find("span", class_="posted-time-ago__text"))

            applicants_tag = (
                job_soup.find("figcaption", class_="num-applicants__caption")
                or job_soup.find("span", class_="num-applicants__caption")
            )
            applicants = safe_text(applicants_tag)

            job_info_lists.append({
                "title": title,
                "organisation_name": organisation_name,
                "organisation_url": organisation_url,
                "description": description,
                "criteria": criteria,
                "url": job_url,
                "posted_time": posted_time,
                "applicants": applicants
            })

    # Save updated list after all queries
    with open("./offers/Jobs_.txt", "w", encoding="utf-8") as f:
        json.dump(job_info_lists, f, ensure_ascii=False, indent=2)

    return job_info_lists
