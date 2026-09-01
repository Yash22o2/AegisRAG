"""
utils.py — Shared utilities for AegisRAG chunking strategies.

Why this file exists:
    Three strategies (fixed_size, semantic, structure_aware) all need the same
    token-counting logic and chunk-ID generation. Keeping them here avoids
    copy-paste drift and ensures that if we switch the tokenizer in the future,
    there is exactly one place to change it.

Functions:
    estimate_tokens(text)          — Approximate token count (words × 1.33)
    make_chunk_id(doc_id, strategy, index) — Deterministic chunk ID string
    split_into_sentences(text)     — Regex sentence splitter (no NLTK dependency)
    find_markdown_headers(text)    — Returns list of (line_index, level, header_text)
                                    for structure_aware strategy
"""

from __future__ import annotations

import re
from typing import List, Tuple


def estimate_tokens(text: str) -> int:
    """
    Approximate token count using word count × 1.33.

    Why not tiktoken: it requires downloading a tokenizer vocabulary file and
    adds a non-trivial import cost. For chunking purposes (where we're making
    rough size decisions, not exact billing calculations), word × 1.33 is a
    standard approximation that's accurate to within ~10% for English text.
    It also avoids adding an OpenAI SDK dependency to a module that should
    work completely offline.

    Args:
        text: Any string.

    Returns:
        Estimated token count as an integer (minimum 1 for non-empty strings).
    """
    if not text or not text.strip():
        return 0
    word_count = len(text.split())
    return max(1, int(word_count * 1.33))


def make_chunk_id(doc_id: str, strategy: str, index: int) -> str:
    """
    Build a deterministic, human-readable chunk ID.

    Format: "{doc_id}_{strategy}_{index:04d}"
    Example: "1a2b3c4d_fixed_size_0007"

    Deterministic IDs matter for Phase 3: when comparing evaluation runs
    across strategies, chunk IDs must be stable across re-runs so that
    stored ground-truth mappings remain valid.

    Args:
        doc_id:   The parent NormalizedDocument's doc_id.
        strategy: One of "fixed_size", "semantic", "structure_aware".
        index:    0-based position within this document's chunk list.

    Returns:
        A string suitable for use as a ChromaDB document ID.
    """
    # ChromaDB IDs can't contain spaces; replace any in doc_id just in case
    safe_doc_id = doc_id.replace(" ", "_")
    return f"{safe_doc_id}_{strategy}_{index:04d}"


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a regex-based approach.

    Why not NLTK/spaCy: Both require downloading models (NLTK punkt tokenizer,
    spaCy language model), which adds setup friction and disk usage. For our
    corpus (well-formed English company documents), a regex approach handles
    ≥95% of cases correctly and has zero external dependencies.

    The regex splits on ". ", "! ", "? " followed by a capital letter or end
    of string. It preserves the sentence-ending punctuation.

    Limitations (acceptable for this corpus):
      - Will mis-split on "Mr. Smith" or "e.g. the policy" — infrequent in
        our documents and the semantic strategy merges neighboring sentences
        anyway, so occasional mis-splits don't degrade retrieval quality.
      - Does not handle ellipses ("...") as a single sentence.

    Args:
        text: Raw document content or a paragraph of text.

    Returns:
        List of sentence strings. Empty strings are filtered out.
    """
    if not text or not text.strip():
        return []

    # Split on sentence-ending punctuation followed by whitespace + capital,
    # or end of string. Using a lookahead so punctuation stays with its sentence.
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def find_markdown_headers(text: str) -> List[Tuple[int, int, str]]:
    """
    Find all Markdown headers in text, returning their line positions.

    Returns a list of tuples: (char_offset, level, header_text)
      - char_offset: character index where this header starts in `text`
      - level: 1 for #, 2 for ##, 3 for ###
      - header_text: the header content without the leading #s and space

    Used by structure_aware.py to identify chunk boundaries.

    Args:
        text: Full document content string.

    Returns:
        Sorted list of (char_offset, level, header_text) tuples.
    """
    headers = []
    # Walk the text line by line, tracking character offset
    offset = 0
    for line in text.split('\n'):
        stripped = line.lstrip()
        if stripped.startswith('### '):
            headers.append((offset, 3, stripped[4:].strip()))
        elif stripped.startswith('## '):
            headers.append((offset, 2, stripped[3:].strip()))
        elif stripped.startswith('# '):
            headers.append((offset, 1, stripped[2:].strip()))
        offset += len(line) + 1  # +1 for the '\n' that split() consumed
    return headers
