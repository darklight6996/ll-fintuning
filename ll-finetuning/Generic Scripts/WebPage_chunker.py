#!/usr/bin/env python3
"""
PortSwigger Web Academy -> RAG TXT Corpus Builder
Extracts learning content, chunks it, and writes to a structured .txt file.
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import hashlib
import argparse
from urllib.parse import urljoin

# Configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RAG-Crawler/1.0"
CHUNK_SIZE_TOKENS = 700  # ~400-500 words
OVERLAP_TOKENS = 80      # ~50-80 words
DELAY_BETWEEN_REQUESTS = 2.0  # Respect rate limits

def fetch_page(url):
    """Fetch HTML with rate limiting and basic error handling."""
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return None

def extract_content(html, url):
    """Extract title and main learning content, stripping noise."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Target semantic learning content
    main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|body|learning", re.I))
    if not main:
        print(f"⚠️ No main content found for {url}")
        return None, None

    # Remove navigation, footer, sidebar, ads, scripts
    for tag in main.find_all(["nav", "footer", "header", "aside", "script", "style", "iframe", "button", "form"]):
        tag.decompose()

    title_el = soup.find("h1") or soup.find("title")
    title = title_el.get_text(strip=True) if title_el else url.split("/")[-1]
    
    # Clean text: remove excessive whitespace
    text = re.sub(r'\s+', ' ', main.get_text()).strip()
    return title, text

def chunk_text(text, title, url, chunk_tokens, overlap_tokens):
    """Split text into overlapping chunks with RAG metadata."""
    if not text or len(text) < 200:
        return []

    # Simple token approximation (1 token ≈ 0.75 words)
    words = text.split()
    chunk_words = int(chunk_tokens / 0.75)
    overlap_words = int(overlap_tokens / 0.75)
    
    chunks = []
    idx = 0
    while idx < len(words):
        end = idx + chunk_words
        chunk_words_list = words[idx:end]
        chunk_text = " ".join(chunk_words_list)
        
        # Metadata
        chunk_id = hashlib.md5(f"{url}_{idx}".encode()).hexdigest()[:8]
        
        chunks.append(
            f"### BEGIN CHUNK ###\n"
            f"# ID: {chunk_id}\n"
            f"# SOURCE: {url}\n"
            f"# TITLE: {title}\n"
            f"# SECTION: Learning Module\n"
            f"### CONTENT ###\n"
            f"{chunk_text}\n"
            f"### END CHUNK ###\n\n"
        )
        
        idx += chunk_words - overlap_words
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Scrape PortSwigger Academy -> RAG TXT")
    parser.add_argument("-i", "--input", required=True, help="Path to .txt file with one URL per line")
    parser.add_argument("-o", "--output", required=True, help="Output RAG corpus .txt file")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_TOKENS)
    parser.add_argument("--overlap", type=int, default=OVERLAP_TOKENS)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        urls = [line.strip() for line in f if line.strip().startswith("http")]

    print(f"🕷️ Processing {len(urls)} URLs...")
    with open(args.output, "w", encoding="utf-8") as out:
        for url in urls:
            print(f"📄 {url}")
            html = fetch_page(url)
            if not html:
                continue
            title, text = extract_content(html, url)
            if not text:
                continue
            chunks = chunk_text(text, title, url, args.chunk_size, args.overlap)
            out.writelines(chunks)
            time.sleep(DELAY_BETWEEN_REQUESTS)
    print(f"✅ RAG corpus saved to {args.output}")

if __name__ == "__main__":
    main()