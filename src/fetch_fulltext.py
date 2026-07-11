import json
import os
import time
import requests
from pypdf import PdfReader
from io import BytesIO

RAW_DIR = "data/raw"

def arxiv_id_to_pdf_url(entry_id):
    # entry_id looks like: http://arxiv.org/abs/2506.08133v1
    arxiv_id = entry_id.split("/abs/")[-1]
    return f"https://arxiv.org/pdf/{arxiv_id}"

def extract_text_from_pdf(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text.strip()

def main():
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".json"))
    print(f"Found {len(files)} paper records")

    for filename in files:
        path = os.path.join(RAW_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            paper = json.load(f)

        if "full_text" in paper:
            continue  # already done, skip

        pdf_url = arxiv_id_to_pdf_url(paper["id"])
        print(f"Fetching: {paper['title'][:60]}...")

        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            full_text = extract_text_from_pdf(response.content)
            paper["full_text"] = full_text
            paper["full_text_chars"] = len(full_text)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(paper, f, indent=2)

        except Exception as e:
            print(f"  FAILED: {e}")
            paper["full_text"] = None
            paper["fetch_error"] = str(e)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(paper, f, indent=2)

        time.sleep(3)  # be polite, PDF downloads are heavier than the API calls

    print("Done.")

if __name__ == "__main__":
    main()
