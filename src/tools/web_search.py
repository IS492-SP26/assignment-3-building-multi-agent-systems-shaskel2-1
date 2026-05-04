"""
Web Search Tool
Connects to Tavily or Brave Search APIs and returns structured results.
Uses synchronous HTTP so it can be safely called inside AutoGen tool wrappers.
"""

import logging
import os
from typing import Any, Dict, List, Optional


logger = logging.getLogger("tools.web_search")


def _search_tavily(query: str, max_results: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set; returning empty results")
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results, search_depth="basic")
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "score": item.get("score", 0.0),
                "published_date": item.get("published_date"),
            })
        return results
    except ImportError:
        logger.error("tavily-python not installed. Run: pip install tavily-python")
        return []
    except Exception as exc:
        logger.error("Tavily search error: %s", exc)
        return []


def _search_brave(query: str, max_results: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        logger.warning("BRAVE_API_KEY not set; returning empty results")
        return []
    try:
        import requests
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }
        params = {"q": query, "count": max_results}
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
                "score": 1.0,
                "published_date": item.get("age"),
            })
        return results
    except Exception as exc:
        logger.error("Brave search error: %s", exc)
        return []


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web and return formatted results.

    Tries Tavily first (if TAVILY_API_KEY is set), then Brave (BRAVE_API_KEY).
    Returns a human-readable string suitable for agent consumption.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)

    Returns:
        Formatted string listing titles, URLs, and snippets.
    """
    results: List[Dict[str, Any]] = []

    if os.getenv("TAVILY_API_KEY"):
        results = _search_tavily(query, max_results)
    elif os.getenv("BRAVE_API_KEY"):
        results = _search_brave(query, max_results)
    else:
        return (
            "Web search is unavailable: no TAVILY_API_KEY or BRAVE_API_KEY found. "
            "Please provide relevant information from your knowledge."
        )

    if not results:
        return f"No web search results found for '{query}'."

    lines = [f"Web search results for '{query}' ({len(results)} results):\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   URL: {r['url']}")
        snippet = r.get("snippet", "").replace("\n", " ").strip()
        if snippet:
            lines.append(f"   {snippet[:300]}")
        if r.get("published_date"):
            lines.append(f"   Published: {r['published_date']}")
        lines.append("")

    return "\n".join(lines)
