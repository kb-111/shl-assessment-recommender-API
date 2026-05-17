"""
SHL catalog scraper — Individual Test Solutions only.

Usage:
    python scripts/scraper.py --output data/shl_catalog.json

Strategy:
  1. Iterate the paginated product catalog with start=0,12,24,...
  2. For each assessment card, extract name, URL, test-type badges, remote/adaptive flags.
  3. Follow the detail URL to extract description and additional metadata.
  4. Retry transient errors with exponential back-off (tenacity).
  5. Output a JSON array of CatalogItem-shaped dicts.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SHLBot/1.0; +https://github.com/shl-intern)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
PAGE_SIZE = 12


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def fetch(url: str, params: dict | None = None) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def parse_test_type(badge_text: str) -> str:
    """Map SHL badge labels to single-char type codes used in API responses."""
    mapping = {
        "ability": "A",
        "biodata": "B",
        "competency": "C",
        "exercise": "E",
        "job focused": "J",
        "knowledge": "K",
        "personality": "P",
        "simulation": "S",
        "skills": "K",          # treat skills tests as knowledge
        "situational": "S",
    }
    lower = badge_text.strip().lower()
    for key, code in mapping.items():
        if key in lower:
            return code
    return "K"  # safe default


def scrape_detail(url: str) -> dict:
    """
    Scrape an individual assessment detail page and return extra metadata.
    Returns a dict with keys: description, duration, job_levels, languages, skills.
    All fields default to empty/None so callers never KeyError.
    """
    detail: dict = {
        "description": "",
        "duration": None,
        "job_levels": [],
        "languages": [],
        "skills": [],
        "remote_testing": False,
        "adaptive": False,
    }
    try:
        soup = fetch(url)

        # Description — usually the first <p> in the main content area
        content = soup.select_one(".product-catalogue-training-calendar__row, .product-detail__content, main article, main .content")
        if content:
            paras = content.find_all("p", limit=3)
            detail["description"] = " ".join(p.get_text(" ", strip=True) for p in paras if p.get_text(strip=True))

        # Fallback description from meta
        if not detail["description"]:
            meta = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
            if meta:
                detail["description"] = meta.get("content", "")

        # Duration
        for label in soup.find_all(string=lambda t: t and "minut" in t.lower()):
            parent = label.parent
            if parent:
                txt = parent.get_text(" ", strip=True)
                if any(c.isdigit() for c in txt):
                    detail["duration"] = txt[:80]
                    break

        # Remote testing / adaptive flags from icon alt text or label
        page_text = soup.get_text(" ", strip=True).lower()
        detail["remote_testing"] = "remote" in page_text
        detail["adaptive"] = "adaptive" in page_text

        # Job levels
        for tag in soup.select(".shl-tag, .badge, .chip"):
            txt = tag.get_text(strip=True).lower()
            if any(w in txt for w in ["entry", "mid", "senior", "executive", "graduate", "professional", "manager"]):
                detail["job_levels"].append(tag.get_text(strip=True))

        # Languages
        for tag in soup.select(".shl-tag, .badge, .chip"):
            txt = tag.get_text(strip=True)
            if len(txt) == 2 and txt.isupper():  # ISO 639-1 codes
                detail["languages"].append(txt)

    except Exception as exc:
        log.warning("Detail scrape failed for %s: %s", url, exc)

    return detail


def scrape_catalog(max_pages: int = 100) -> list[dict]:
    """
    Iterate all paginated pages of the SHL product catalog.
    Filters to Individual Test Solutions only (excludes Pre-packaged Job Solutions).
    Returns list of raw dicts ready to write as JSON.
    """
    items: list[dict] = []
    seen_urls: set[str] = set()

    for page_idx in range(max_pages):
        start = page_idx * PAGE_SIZE
        log.info("Scraping page %d (start=%d)…", page_idx + 1, start)

        try:
            soup = fetch(CATALOG_URL, params={"start": start, "type": "1"})
        except Exception as exc:
            log.error("Failed to fetch catalog page %d: %s", page_idx + 1, exc)
            break

        # Find product cards — SHL uses a table or card grid
        cards = soup.select("tr.product-catalogue-training-calendar__row, .custom-select__item, [data-course-id]")
        
        # Fallback: find all links inside product listing area
        if not cards:
            listing = soup.select_one(
                ".product-catalogue, #product-catalogue, .solutions-list, main table"
            )
            if listing:
                cards = listing.select("tr, .item, li")

        # Another fallback: parse table rows directly
        if not cards:
            cards = soup.select("table tr")

        if not cards:
            log.info("No cards found on page %d — end of catalog.", page_idx + 1)
            break

        page_items = 0
        for card in cards:
            # Extract anchor with the assessment link
            anchor = card.select_one("a[href*='/solutions/products/'], a[href*='product-catalog']")
            if not anchor:
                continue
            href = anchor.get("href", "")
            if not href:
                continue
            full_url = href if href.startswith("http") else BASE_URL + href

            # Skip if already seen or if it's a Job Solution (pre-packaged)
            if full_url in seen_urls:
                continue
            if "job-solution" in full_url.lower() or "pre-packaged" in full_url.lower():
                continue
            seen_urls.add(full_url)

            name = anchor.get_text(" ", strip=True) or card.get_text(" ", strip=True)[:80]
            if not name.strip():
                continue

            # Extract test type badges from the card row
            badges = card.select(".badge, .product-catalogue__key, .shl-tag, td:nth-child(2)")
            test_type = "K"
            for badge in badges:
                txt = badge.get_text(strip=True)
                if txt:
                    test_type = parse_test_type(txt)
                    break

            # Remote / adaptive icons — look for checkmarks or specific columns
            remote = False
            adaptive = False
            tds = card.select("td")
            if len(tds) >= 4:
                # Columns: Name | Test Type | Remote | Adaptive (typical SHL table)
                remote_td = tds[3].get_text(strip=True).lower() if len(tds) > 3 else ""
                adaptive_td = tds[4].get_text(strip=True).lower() if len(tds) > 4 else ""
                remote = bool(remote_td) and remote_td not in ("no", "–", "-", "")
                adaptive = bool(adaptive_td) and adaptive_td not in ("no", "–", "-", "")

            item: dict = {
                "name": name.strip(),
                "url": full_url,
                "test_type": test_type,
                "description": "",
                "duration": None,
                "job_levels": [],
                "languages": [],
                "skills": [],
                "remote_testing": remote,
                "adaptive": adaptive,
            }

            # Scrape detail page for richer metadata (rate-limited)
            detail = scrape_detail(full_url)
            item.update(detail)

            items.append(item)
            page_items += 1
            log.info("  + %s [%s]", name[:60], test_type)
            time.sleep(0.5)  # polite crawl delay

        if page_items == 0:
            log.info("Empty page — stopping.")
            break

    log.info("Scraped %d assessments total.", len(items))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape SHL product catalog")
    parser.add_argument("--output", default="data/shl_catalog.json")
    parser.add_argument("--max-pages", type=int, default=50)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    items = scrape_catalog(max_pages=args.max_pages)

    if not items:
        log.error("No items scraped. Check network access and selectors.")
        return

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    log.info("Saved %d items to %s", len(items), args.output)


if __name__ == "__main__":
    main()