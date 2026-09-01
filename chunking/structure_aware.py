"""
structure_aware.py — Structure-aware (hierarchical) chunking strategy for AegisRAG.

Why this strategy exists:
    Both fixed-size and semantic chunking are "content-blind" in the sense that
    they don't understand what a heading is. Structure-aware chunking treats
    Markdown headings as hard chunk boundaries — because they ARE hard semantic
    boundaries: every heading in the corpus marks a genuine topic change. This
    is especially powerful for our Notion docs (Engineering Runbook, IT Security
    Policy, etc.) which have rich heading hierarchies, and less powerful for
    messy docs (meeting notes) which have almost no headings.

    This variance is intentional: Phase 3's evaluation harness should show
    structure-aware outperforming fixed-size and semantic on structured Notion docs
    but performing comparably on unstructured Drive docs — that's the genuine,
    data-backed finding that makes this project's eval story coherent.

    The metadata.section_header field (populated only by this strategy) is also
    directly useful in Phase 4: the RAG generator can format citations as
    "From: Engineering Runbook > On-Call Rotation" instead of just a bare URL,
    which is a concrete quality improvement reviewers can see.

Algorithm:
    1. Parse document content for Markdown headers (# / ## / ###) using
       utils.find_markdown_headers()
    2. Each header starts a new segment. Content before the first header
       (if any) forms a "preamble" segment with section_header=None.
    3. For each segment:
       a. If token count <= max_tokens: emit as one Chunk, section_header = the
          heading text above it.
       b. If token count > max_tokens: sub-split with fixed-size sliding window
          (same logic as fixed_chunker). Each sub-chunk retains the parent
          section_header so citations still work.
    4. If no headers found at all (e.g., meeting notes): fall back to fixed-size
       chunking of the entire document (emit one or more Chunks with
       section_header=None). Log this fallback.

Public API:
    chunk_structure_aware(doc, max_tokens=512) -> list[Chunk]

Parameters:
    max_tokens: Maximum chunk size in tokens (default 512). Sections that exceed
                this are sub-split. Default matches fixed_chunker for fair comparison.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from mcp_server.normalize import NormalizedDocument
from chunking.schema import Chunk, make_chunk
from chunking.utils import find_markdown_headers, estimate_tokens


def _words_window(
    text: str,
    window_words: int,
    step_words: int,
    doc_id: str,
    source: str,
    strategy: str,
    title: str,
    doc_type: str,
    path_or_url: Optional[str],
    section_header: Optional[str],
    start_char_offset: int,
    chunk_index_start: int,
) -> List[dict]:
    """
    Sub-split `text` with a fixed-size word window.

    Returns raw dicts (not Chunk objects) so the caller can assign final
    chunk_index values. Each dict has keys: text, start_char, end_char,
    section_header (inherited from parent segment).
    """
    words = text.split()
    sub_chunks = []
    pos = 0
    while pos < len(words):
        end_pos = min(pos + window_words, len(words))
        chunk_text = " ".join(words[pos:end_pos])
        # Approximate char offsets within the segment
        approx_start = start_char_offset + len(" ".join(words[:pos])) + (1 if pos > 0 else 0)
        approx_end = approx_start + len(chunk_text)
        sub_chunks.append({
            "text": chunk_text,
            "start_char": approx_start,
            "end_char": approx_end,
            "section_header": section_header,
        })
        if end_pos >= len(words):
            break
        pos += step_words
    return sub_chunks


def chunk_structure_aware(
    doc: NormalizedDocument,
    max_tokens: int = 512,
) -> List[Chunk]:
    """
    Split a NormalizedDocument into chunks aligned with Markdown structure.

    Args:
        doc:        A NormalizedDocument (from any MCP connector).
        max_tokens: Maximum chunk size in tokens (default 512). Sections larger
                    than this are sub-split with fixed-size word windows.

    Returns:
        List of Chunk objects with metadata.section_header populated where
        the document has Markdown headings.
    """
    content = doc.content
    if not content or not content.strip():
        return []

    headers = find_markdown_headers(content)

    # -----------------------------------------------------------------------
    # Fallback: no headers found — run fixed-size chunking instead
    # -----------------------------------------------------------------------
    if not headers:
        print(
            f"[structure_aware] No Markdown headers in '{doc.title}' "
            f"({doc.doc_id}) — falling back to fixed-size chunking.",
            file=sys.stderr,
        )
        from chunking.fixed_chunker import chunk_fixed
        # Re-tag the returned chunks as structure_aware strategy and return
        fixed_chunks = chunk_fixed(doc, chunk_size=max_tokens, overlap=64)
        result = []
        for idx, c in enumerate(fixed_chunks):
            result.append(make_chunk(
                doc_id=c.doc_id,
                source=c.source,
                strategy="structure_aware",
                text=c.text,
                chunk_index=idx,
                title=c.metadata.title,
                doc_type=c.metadata.doc_type,
                path_or_url=c.metadata.path_or_url,
                start_char=c.metadata.start_char,
                end_char=c.metadata.end_char,
                section_header=None,  # No headers to track
            ))
        return result

    # -----------------------------------------------------------------------
    # Normal path: use header positions as segment boundaries
    # -----------------------------------------------------------------------

    # Build segment list: each segment is (start_char, end_char, section_header)
    # Content before the first header (if any) is a preamble segment.
    segments: List[Tuple[int, int, Optional[str]]] = []

    # Preamble: content before the first header
    first_header_offset = headers[0][0]
    if first_header_offset > 0:
        preamble = content[:first_header_offset].strip()
        if preamble:
            segments.append((0, first_header_offset, None))

    # Each header → next header (or end of content)
    for i, (offset, level, header_text) in enumerate(headers):
        # Find where this segment ends: start of the next header, or EOF
        if i + 1 < len(headers):
            next_offset = headers[i + 1][0]
        else:
            next_offset = len(content)

        # Skip the header line itself; segment content starts after it
        # Find the end of the header line
        newline_pos = content.find('\n', offset)
        if newline_pos == -1 or newline_pos >= next_offset:
            # Single-line header at end of content or no newline
            seg_content_start = next_offset
        else:
            seg_content_start = newline_pos + 1

        segments.append((seg_content_start, next_offset, header_text))

    # -----------------------------------------------------------------------
    # Convert segments to Chunk objects
    # -----------------------------------------------------------------------
    window_words = max(1, int(max_tokens / 1.33))
    step_words = max(1, window_words - int(64 / 1.33))  # 64-token overlap

    chunks: List[Chunk] = []
    chunk_index = 0

    for seg_start, seg_end, section_header in segments:
        seg_text = content[seg_start:seg_end].strip()
        if not seg_text:
            continue

        seg_tokens = estimate_tokens(seg_text)

        if seg_tokens <= max_tokens:
            # Segment fits in one chunk — emit directly
            chunks.append(make_chunk(
                doc_id=doc.doc_id,
                source=doc.source,
                strategy="structure_aware",
                text=seg_text,
                chunk_index=chunk_index,
                title=doc.title,
                doc_type=doc.metadata.doc_type,
                path_or_url=doc.metadata.path_or_url,
                start_char=seg_start,
                end_char=seg_end,
                section_header=section_header,
            ))
            chunk_index += 1
        else:
            # Segment too large — sub-split with fixed-size, but preserve header
            sub_parts = _words_window(
                seg_text,
                window_words=window_words,
                step_words=step_words,
                doc_id=doc.doc_id,
                source=doc.source,
                strategy="structure_aware",
                title=doc.title,
                doc_type=doc.metadata.doc_type,
                path_or_url=doc.metadata.path_or_url,
                section_header=section_header,
                start_char_offset=seg_start,
                chunk_index_start=chunk_index,
            )
            for part in sub_parts:
                chunks.append(make_chunk(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    strategy="structure_aware",
                    text=part["text"],
                    chunk_index=chunk_index,
                    title=doc.title,
                    doc_type=doc.metadata.doc_type,
                    path_or_url=doc.metadata.path_or_url,
                    start_char=part["start_char"],
                    end_char=part["end_char"],
                    section_header=part["section_header"],  # Inherited!
                ))
                chunk_index += 1

    return chunks
