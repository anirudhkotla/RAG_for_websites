import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from chunking import structured_chunk
from embeddings import embed_chunks, save_index

def fetch_html(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

def parse_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator="\n")

def fetch_sitemap_urls(sitemap_url):
    try:
        resp = requests.get(sitemap_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        urls = [loc.text for loc in soup.find_all("loc")]
        print(f"Found {len(urls)} URLs in sitemap")
        return urls
    except Exception as e:
        print(f"Sitemap fetch failed: {e}")
        return []

def crawl_website(base_url, max_pages=50):
    visited = set()
    queue = deque([base_url])
    all_urls = []

    while queue and len(all_urls) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        html = fetch_html(url)
        if not html:
            continue

        all_urls.append(url)
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            abs_link = urljoin(base_url, link['href'])
            if urlparse(abs_link).netloc == urlparse(base_url).netloc:
                queue.append(abs_link)

    print(f"Crawler found {len(all_urls)} pages")
    return all_urls

def ingest_url(base_url, sitemap_url=None):
    urls_to_fetch = crawl_website(base_url)
    if sitemap_url:
        sitemap_urls = fetch_sitemap_urls(sitemap_url)
        # Add sitemap URLs only if not already found by crawler
        for url in sitemap_urls:
            if url not in urls_to_fetch:
                urls_to_fetch.append(url)
        print(f"Total URLs after merging crawler + sitemap: {len(urls_to_fetch)}")

    all_texts = []
    for url in urls_to_fetch:
        html = fetch_html(url)
        text = parse_html(html)
        all_texts.append(text)

    full_text = "\n".join(all_texts)
    chunks = structured_chunk(full_text)
    embeddings = embed_chunks(chunks)
    save_index(chunks, embeddings)
    print("Ingestion complete.")