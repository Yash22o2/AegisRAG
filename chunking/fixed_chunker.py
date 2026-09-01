"""
fixed_chunker.py — Fixed-size (sliding window) chunking strategy for AegisRAG.

Why this strategy exists (and why it's the baseline):
    Fixed-size chunking is the simplest possible approach: split the document
    into word-count windows of a fixed size with a fixed overlap. It completely
    ignores document structure — headings, paragraphs, sentence boundaries are
    all irrelevant. This makes it the perfect baseline because:

    1. It's reproducible — same parameters always produce the same chunks.
    2. It's the most common "naive" approach in RAG tutorials, so reviewers
       immediately understand what it does and what its weaknesses are.
    3. Its weaknesses are predictable: it will split mid-sentence and mid-section,
       which should show up as lower Context Precision in Phase 3 for structured
       docs (Notion policy pages) — and that gap is the story the eval harness
       tells.

Algorithm:
    1. Split content into a flat list of words (whitespace tokenization).
    2. Slide a window of `chunk_size` words, advancing by (chunk_size - overlap)
       words each step.
    3. For each window, rejoin words into a text string and record the character
       offsets by scanning the original text for the first word of the window
       (approximate — exact offsets require character-level tracking which adds
       complexity not justified for this strategy's use case).
    4. Produce one Chunk per window.

Public API:
    chunk_fixed(doc, chunk_size=512, overlap=64) -> list[Chunk]

Parameters:
    chunk_size: Target chunk size in estimated tokens (default 512).
                Since we use word × 1.33 for token estimation, 512 tokens ≈
                385 words. The sliding window uses word count directly.
    overlap:    Number of tokens to overlap between consecutive chunks (default 64).
                Overlap ensures that content near chunk boundaries appears in
                two consecutive chunks, reducing the chance of a query falling
                entirely in the gap between them.
"""

from __future__ import annotations

import math
from typing import List

from mcp_server.normalize import NormalizedDocument
from chunking.schema import Chunk, make_chunk
from chunking.utils import estimate_tokens


# Default parameters — chosen to match common RAG benchmarks.
# chunk_size=512 tokens ≈ 385 words; overlap=64 tokens ≈ 48 words.
_DEFAULT_CHUNK_SIZE_TOKENS = 512
_DEFAULT_OVERLAP_TOKENS = 64

# Token-to-word conversion factor (word × 1.33 ≈ tokens)
_WORDS_PER_TOKEN = 1 / 1.33  # ≈ 0.75


def _tokens_to_words(tokens: int) -> int:
    """Convert an approximate token count to a word count."""
    return max(1, int(tokens * _WORDS_PER_TOKEN))


def chunk_fixed(
    doc: NormalizedDocument,
    chunk_size: int = _DEFAULT_CHUNK_SIZE_TOKENS,
    overlap: int = _DEFAULT_OVERLAP_TOKENS,
) -> List[Chunk]:
    """
    Split a NormalizedDocument into fixed-size overlapping chunks.

    Args:
        doc:        A NormalizedDocument (from any MCP connector).
        chunk_size: Target chunk size in tokens (default 512).
        overlap:    Overlap between consecutive chunks in tokens (default 64).

    Returns:
        List of Chunk objects. Empty list if doc.content is empty or whitespace.

    Note:
        Character offsets (start_char / end_char) in ChunkMetadata are
        approximated by scanning the original text for the rejoined chunk text.
        This is accurate enough for traceability but not byte-perfect.
    """
    content = doc.content
    if not content or not content.strip():
        return []

    # Convert token counts to word counts for the sliding window
    window_words = _tokens_to_words(chunk_size)
    step_words = max(1, _tokens_to_words(chunk_size - overlap))

    # Tokenize by whitespace — preserves punctuation attached to words,
    # which is fine since we're rejoining them anyway.
    words = content.split()

    if not words:
        return []

    chunks: List[Chunk] = []
    chunk_index = 0
    pos = 0  # Current word position

    while pos < len(words):
        end_pos = min(pos + window_words, len(words))
        window = words[pos:end_pos]
        chunk_text = " ".join(window)

        # Approximate character offsets: find where this chunk text appears
        # starting from a rough position. This is a best-effort scan —
        # exact offsets would require character-level tokenization.
        approx_start = len(" ".join(words[:pos]))
        if pos > 0:
            approx_start += 1  # for the space separator
        approx_end = approx_start + len(chunk_text)

        chunk = make_chunk(
            doc_id=doc.doc_id,
            source=doc.source,
            strategy="fixed_size",
            text=chunk_text,
            chunk_index=chunk_index,
            title=doc.title,
            doc_type=doc.metadata.doc_type,
            path_or_url=doc.metadata.path_or_url,
            start_char=approx_start,
            end_char=approx_end,
            section_header=None,  # Fixed-size does not track structure
        )
        chunks.append(chunk)
        chunk_index += 1

        # If we've reached the end, stop
        if end_pos >= len(words):
            break

        pos += step_words

    return chunks


def chunk_fixed_summary(chunks: List[Chunk]) -> dict:
    """
    Return a summary dict for logging/display after chunking.

    Useful for the ingest.py driver script and spot-check verification.
    """
    if not chunks:
        return {"count": 0, "avg_tokens": 0, "min_tokens": 0, "max_tokens": 0}
    token_counts = [c.token_count for c in chunks]
    return {
        "count": len(chunks),
        "avg_tokens": round(sum(token_counts) / len(token_counts)),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
    }
