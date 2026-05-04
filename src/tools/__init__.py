"""
Research Tools Module
"""

from .web_search import web_search
from .paper_search import paper_search
from .citation_tool import CitationTool

__all__ = [
    "web_search",
    "paper_search",
    "CitationTool",
]
