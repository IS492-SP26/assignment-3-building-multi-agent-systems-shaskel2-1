"""
Paper Search Tool
Connects to Semantic Scholar API and returns academic paper metadata.
Uses the synchronous semanticscholar client.
"""

import logging
import os
from typing import Any, Dict, List, Optional


logger = logging.getLogger("tools.paper_search")


def paper_search(query: str, max_results: int = 8, year_from: Optional[int] = None) -> str:
    """
    Search Semantic Scholar for academic papers and return formatted results.

    Args:
        query: Search query string
        max_results: Maximum number of papers to return (default 8)
        year_from: If provided, only return papers published from this year onwards

    Returns:
        Formatted string listing paper titles, authors, years, abstracts, and URLs.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    try:
        from semanticscholar import SemanticScholar

        sch = SemanticScholar(api_key=api_key if api_key else None)

        fields = ["paperId", "title", "authors", "year", "abstract",
                  "citationCount", "url", "venue", "openAccessPdf"]

        raw_results = sch.search_paper(query, limit=max_results, fields=fields)

        papers: List[Dict[str, Any]] = []
        for paper in raw_results:
            if not paper or not getattr(paper, "title", None):
                continue
            p = {
                "title": paper.title or "Unknown",
                "authors": [{"name": a.name} for a in (paper.authors or [])],
                "year": getattr(paper, "year", None),
                "abstract": getattr(paper, "abstract", "") or "",
                "citation_count": getattr(paper, "citationCount", 0) or 0,
                "url": getattr(paper, "url", "") or "",
                "venue": getattr(paper, "venue", "") or "",
                "pdf_url": (
                    paper.openAccessPdf.get("url")
                    if getattr(paper, "openAccessPdf", None)
                    else None
                ),
            }
            if year_from and p["year"] and p["year"] < year_from:
                continue
            papers.append(p)
            if len(papers) >= max_results:
                break

        if not papers:
            return f"No academic papers found for '{query}'."

        lines = [f"Academic papers for '{query}' ({len(papers)} results):\n"]
        for i, p in enumerate(papers, 1):
            authors = ", ".join(a["name"] for a in p["authors"][:3])
            if len(p["authors"]) > 3:
                authors += " et al."

            lines.append(f"{i}. {p['title']}")
            lines.append(f"   Authors: {authors or 'Unknown'}")
            year_str = str(p["year"]) if p["year"] else "n.d."
            lines.append(f"   Year: {year_str} | Citations: {p['citation_count']}" +
                         (f" | Venue: {p['venue']}" if p["venue"] else ""))
            if p["abstract"]:
                snippet = p["abstract"][:250].replace("\n", " ")
                lines.append(f"   Abstract: {snippet}...")
            if p["url"]:
                lines.append(f"   URL: {p['url']}")
            lines.append("")

        return "\n".join(lines)

    except ImportError:
        logger.error("semanticscholar not installed. Run: pip install semanticscholar")
        return "Paper search unavailable: semanticscholar library not installed."
    except Exception as exc:
        logger.error("Paper search error: %s", exc)
        return f"Paper search encountered an error: {exc}. Please rely on web search instead."
