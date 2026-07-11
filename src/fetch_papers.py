import requests
import feedparser
import json
import os
import time

# --- CONFIG ---
QUERY = "all:retrieval augmented generation"  # change this to your topic if you want
MAX_RESULTS = 60
OUTPUT_DIR = "data/raw"
BASE_URL = "http://export.arxiv.org/api/query"

def fetch_papers(query, max_results):
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return feedparser.parse(response.text)

def save_paper(entry, index):
    paper = {
        "id": entry.id,
        "title": entry.title.strip(),
        "authors": [a.name for a in entry.authors],
        "published": entry.published,
        "summary": entry.summary.strip(),
    }
    filename = f"{OUTPUT_DIR}/paper_{index:03d}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(paper, f, indent=2)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    feed = fetch_papers(QUERY, MAX_RESULTS)
    print(f"Found {len(feed.entries)} papers")

    for i, entry in enumerate(feed.entries):
        save_paper(entry, i)
        time.sleep(0.1)

    print(f"Saved {len(feed.entries)} papers to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
