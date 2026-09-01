"""
schema.py — Chunk dataclass for AegisRAG Phase 2.

Why this file exists:
    All three chunking strategies (fixed_size, semantic, structure_aware) must
    produce the same output type so the downstream ingestion pipeline (ingest.py)
    and evaluation harness (Phase 3) don't need to know which strategy produced
    a chunk — they just consume Chunk objects.

    Mirrors the NormalizedDocument / DocumentMetadata pattern from
    mcp_server/normalize.py exactly: dataclasses + a factory function,
    keyword-only arguments, .to_dict() via asdict().

Chunk fields:
    chunk_id        — Deterministic: "{doc_id}_{strategy}_{index:04d}"
    doc_id          — Parent document's ID (links back to NormalizedDocument)
    source          — "notion" | "drive" — inherited from parent doc
    strategy        — "fixed_size" | "semantic" | "structure_aware"
    text            — The actual chunk text
    token_count     — Estimated token count (word-count x 1.33 approximation)
    chunk_index     — 0-based position within this document's chunk list
    metadata        — ChunkMetadata (see below)

ChunkMetadata fields:
    title           — Parent doc title (for display / citation generation)
    doc_type        — Inherited from NormalizedDocument.metadata.doc_type
    path_or_url     — Link back to source (Notion URL or Drive web-view link)
    start_char      — Character offset in original content (for traceability)
    end_char        — Character offset end
    section_header  — Nearest Markdown heading above this chunk; None for
                      fixed_size and semantic (they don't track structure).
                      Populated by structure_aware strategy — used in Phase 4
                      for citation formatting ("From: Engineering Runbook > On-Call Rotation")
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal, Optional


@dataclass
class ChunkMetadata:
    """Source-tracing metadata attached to every Chunk."""

    title: str                         # Parent document title
    doc_type: str                      # "policy" | "runbook" | "meeting_notes" | etc.
    path_or_url: Optional[str]         # Link to original source
    start_char: int                    # Start character offset in original content
    end_char: int                      # End character offset in original content
    section_header: Optional[str]      # Nearest Markdown heading (structure_aware only)


@dataclass
class Chunk:
    """
    A single chunk produced by any of the three chunking strategies.

    Using a dataclass (not a plain dict) gives:
      - Type checking during development
      - .to_dict() for JSON serialization and ChromaDB metadata storage
      - Clear field documentation here in the class
    """

    chunk_id: str
    doc_id: str
    source: Literal["notion", "drive"]
    strategy: Literal["fixed_size", "semantic", "structure_aware"]
    text: str
    token_count: int
    chunk_index: int
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        """Return a fully JSON-serializable plain dict."""
        return asdict(self)


def make_chunk(
    *,
    doc_id: str,
    source: Literal["notion", "drive"],
    strategy: Literal["fixed_size", "semantic", "structure_aware"],
    text: str,
    chunk_index: int,
    title: str,
    doc_type: str,
    path_or_url: Optional[str] = None,
    start_char: int = 0,
    end_char: int = 0,
    section_header: Optional[str] = None,
) -> Chunk:
    """
    Factory function — the only public API callers need.

    Keyword-only arguments (enforced by bare `*`) prevent positional-arg
    ordering mistakes as the signature grows.

    chunk_id is built deterministically from doc_id + strategy + index so
    that re-running ingest.py produces the same IDs (important for reproducible
    eval in Phase 3).
    """
    from chunking.utils import estimate_tokens, make_chunk_id

    return Chunk(
        chunk_id=make_chunk_id(doc_id, strategy, chunk_index),
        doc_id=doc_id,
        source=source,
        strategy=strategy,
        text=text,
        token_count=estimate_tokens(text),
        chunk_index=chunk_index,
        metadata=ChunkMetadata(
            title=title,
            doc_type=doc_type,
            path_or_url=path_or_url,
            start_char=start_char,
            end_char=end_char,
            section_header=section_header,
        ),
    )
