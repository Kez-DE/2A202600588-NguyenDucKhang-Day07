from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

FIRECRAWL_URL = "http://localhost:3002"
OUTPUT_DIR = Path("data/python_official_docs")
METADATA_PATH = Path("data/python_official_docs_metadata.json")

DOCS = [
    {
        "doc_id": "python_data_structures",
        "filename": "python_data_structures.md",
        "url": "https://docs.python.org/3/tutorial/datastructures.html",
        "title": "Data Structures",
        "topic": "data_structures",
        "difficulty": "beginner",
        "keywords": ["list", "dict", "tuple", "set", "comprehension"],
    },
    {
        "doc_id": "python_modules",
        "filename": "python_modules.md",
        "url": "https://docs.python.org/3/tutorial/modules.html",
        "title": "Modules",
        "topic": "modules",
        "difficulty": "beginner",
        "keywords": ["import", "module", "package", "namespace"],
    },
    {
        "doc_id": "python_errors_exceptions",
        "filename": "python_errors_exceptions.md",
        "url": "https://docs.python.org/3/tutorial/errors.html",
        "title": "Errors and Exceptions",
        "topic": "errors",
        "difficulty": "intermediate",
        "keywords": ["exception", "try", "except", "raise", "finally"],
    },
    {
        "doc_id": "python_classes",
        "filename": "python_classes.md",
        "url": "https://docs.python.org/3/tutorial/classes.html",
        "title": "Classes",
        "topic": "oop",
        "difficulty": "intermediate",
        "keywords": ["class", "object", "instance", "inheritance", "method"],
    },
    {
        "doc_id": "python_stdlib",
        "filename": "python_stdlib.md",
        "url": "https://docs.python.org/3/tutorial/stdlib.html",
        "title": "Brief Tour of the Standard Library",
        "topic": "standard_library",
        "difficulty": "intermediate",
        "keywords": ["os", "sys", "re", "math", "random", "urllib"],
    },
    {
        "doc_id": "python_venv",
        "filename": "python_venv.md",
        "url": "https://docs.python.org/3/tutorial/venv.html",
        "title": "Virtual Environments and Packages",
        "topic": "environment",
        "difficulty": "beginner",
        "keywords": ["venv", "pip", "dependency", "package", "environment"],
    },
    {
        "doc_id": "python_input_output",
        "filename": "python_input_output.md",
        "url": "https://docs.python.org/3/tutorial/inputoutput.html",
        "title": "Input and Output",
        "topic": "io",
        "difficulty": "beginner",
        "keywords": ["file", "open", "read", "write", "format"],
    },
    {
        "doc_id": "python_argparse",
        "filename": "python_argparse.md",
        "url": "https://docs.python.org/3/library/argparse.html",
        "title": "argparse — Parser for command-line options, arguments and subcommands",
        "topic": "cli",
        "difficulty": "intermediate",
        "keywords": ["cli", "argument", "parser", "command-line", "subcommand"],
    },
]


def check_firecrawl() -> None:
    response = requests.get(FIRECRAWL_URL, timeout=10)
    response.raise_for_status()
    message = response.json().get("message")
    if message != "Firecrawl API":
        raise RuntimeError(f"Unexpected Firecrawl response: {response.text[:200]}")


def firecrawl_probe(url: str) -> dict:
    response = requests.post(f"{FIRECRAWL_URL}/v1/scrape", json={"url": url}, timeout=60)
    response.raise_for_status()
    data = response.json()
    markdown_len = len(data.get("data", {}).get("markdown", ""))
    return {"success": bool(data.get("success")), "markdown_length": markdown_len}


def extract_main_markdown(url: str) -> tuple[str, list[str], str]:
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    main = soup.find("main") or soup.find("div", {"role": "main"}) or soup.find("article")
    if main is None:
        raise RuntimeError(f"Could not locate main content for {url}")

    for selector in ["script", "style", "nav", ".headerlink", ".visually-hidden", ".sidebar-drawer", ".toc-drawer"]:
        for node in main.select(selector):
            node.decompose()

    title_node = main.find(["h1", "h2"])
    detected_title = title_node.get_text(" ", strip=True) if title_node else "Untitled"
    headings = [h.get_text(" ", strip=True).replace("¶", "") for h in main.find_all(["h1", "h2", "h3"])]

    markdown = md(str(main), heading_style="ATX", bullets="-")
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(r"¶", "", markdown)
    markdown = markdown.strip()
    return markdown, headings, detected_title


def write_docs() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    scraped_at = datetime.now(timezone.utc).isoformat()

    for doc in DOCS:
        probe = firecrawl_probe(doc["url"])
        markdown, headings, detected_title = extract_main_markdown(doc["url"])
        output_path = OUTPUT_DIR / doc["filename"]

        frontmatter = {
            "doc_id": doc["doc_id"],
            "source_url": doc["url"],
            "source_name": "Python Official Documentation",
            "doc_title": doc["title"],
            "detected_title": detected_title,
            "topic": doc["topic"],
            "difficulty": doc["difficulty"],
            "doc_type": "official_documentation",
            "python_version": "3",
            "language": "en",
            "keywords": doc["keywords"],
            "scraped_at": scraped_at,
            "scraper": "Firecrawl /v1/scrape probe + requests/BeautifulSoup clean extraction",
        }
        file_text = "---\n" + json.dumps(frontmatter, ensure_ascii=False, indent=2) + "\n---\n\n" + markdown + "\n"
        output_path.write_text(file_text, encoding="utf-8")

        records.append(
            {
                **frontmatter,
                "file_path": str(output_path),
                "character_count": len(file_text),
                "heading_count": len(headings),
                "sample_headings": headings[:8],
                "firecrawl_success": probe["success"],
                "firecrawl_markdown_length": probe["markdown_length"],
            }
        )

    METADATA_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records


def main() -> int:
    check_firecrawl()
    records = write_docs()
    print(f"Wrote {len(records)} Python official documentation files to {OUTPUT_DIR}")
    print(f"Wrote metadata inventory to {METADATA_PATH}")
    for record in records:
        print(
            f"- {record['doc_id']}: {record['character_count']} chars, "
            f"{record['heading_count']} headings, firecrawl={record['firecrawl_success']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
