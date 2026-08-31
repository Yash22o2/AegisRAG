"""
normalize.py — Common document schema for AegisRAG MCP server.

Why this file exists:
    Both connectors (Notion and Drive) return raw API responses in completely
    different shapes. All downstream phases (Phase 2 chunking, Phase 3 eval)
    must never need to know or care which source a document came from — they
    should see one uniform schema. This module defines that schema and provides
    the factory function that produces it, so the connectors only need to call
    `make_document(...)` rather than building the dict themselves.

Schema fields:
    doc_id        — Unique identifier: Notion page ID or Drive file ID.
    source        — "notion" | "drive" (literal string, not an enum, to stay
                    JSON-serializable without extra machinery).
    title         — Human-readable title as returned by the API.
    content       — Full extracted plain text of the document.
    metadata      — Nested dict with author, modified_date, doc_type,
                    path_or_url. None-safe: any field may be None if the API
                    didn't return it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal, Optional


@dataclass
class DocumentMetadata:
    """Nested metadata block inside a NormalizedDocument."""

    author: Optional[str]
    modified_date: Optional[str]   # ISO 8601 string, e.g. "2024-07-15T10:30:00Z"
    doc_type: str                  # "policy" | "meeting_notes" | "spec" | etc.
    path_or_url: Optional[str]     # Notion URL or Drive web-view link


@dataclass
class NormalizedDocument:
    """
    The single document shape used throughout AegisRAG.

    This is what every MCP tool returns and what Phase 2 (chunking) consumes.
    Using a dataclass rather than a plain dict gives us:
      - type checking during development
      - .to_dict() via asdict() for JSON serialization
      - clear field documentation right here in the class
    """

    doc_id: str
    source: Literal["notion", "drive"]
    title: str
    content: str
    metadata: DocumentMetadata

    def to_dict(self) -> dict:
        """Return a plain dict (fully JSON-serializable)."""
        return asdict(self)


def make_document(
    *,
    doc_id: str,
    source: Literal["notion", "drive"],
    title: str,
    content: str,
    author: Optional[str] = None,
    modified_date: Optional[str] = None,
    doc_type: str = "unknown",
    path_or_url: Optional[str] = None,
) -> NormalizedDocument:
    """
    Factory function — the only public API callers need.

    Keyword-only arguments (enforced by the bare `*`) prevent the easy mistake
    of passing positional args in the wrong order when the signature grows.

    Usage example (from notion_connector.py):
        doc = make_document(
            doc_id=page["id"],
            source="notion",
            title=extract_title(page),
            content=page_text,
            modified_date=page["last_edited_time"],
            doc_type="policy",
            path_or_url=page["url"],
        )
    """
    metadata = DocumentMetadata(
        author=author,
        modified_date=modified_date,
        doc_type=doc_type,
        path_or_url=path_or_url,
    )
    return NormalizedDocument(
        doc_id=doc_id,
        source=source,
        title=title,
        content=content,
        metadata=metadata,
    )
