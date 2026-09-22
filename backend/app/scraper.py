"""
URL Scraper Module.

Extracts text content from internship posting URLs
using requests and BeautifulSoup (headless).
"""

import requests
from bs4 import BeautifulSoup


def scrape_url(url: str, timeout: int = 15) -> str:
    """
    Scrape text content from a given URL.

    Args:
        url: The URL to scrape.
        timeout: Request timeout in seconds.

    Returns:
        Extracted text content from the page.

    Raises:
        ValueError: If the URL is invalid or unreachable.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch URL: {e}") from e

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Extract text
    text = soup.get_text(separator=" ", strip=True)

    if not text.strip():
        raise ValueError("No text content found at the given URL.")

    return text
