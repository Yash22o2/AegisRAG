"""
semantic.py — Semantic (embedding-similarity) chunking strategy for AegisRAG.

Why this strategy exists:
    Fixed-size chunking splits at arbitrary word boundaries. Semantic chunking
    splits where meaning changes. The core insight: if two adjacent sentences
    have high embedding cosine-similarity, they're talking about the same topic
    and should stay in the same chunk. If similarity drops below a threshold,
    that's a natural topic boundary.

    This strategy should outperform fixed-size on documents where topic shifts
    align with natural boundaries (e.g., the IT Security Policy transitions
    from "Password Policy" to "Device Security" — a sharp semantic shift).
    It should perform similarly to fixed-size on documents with no clear topic
    structure (meeting notes with scattered bullet points).

Algorithm:
    1. Split content into sentences using utils.split_into_sentences()
    2. Embed each sentence with sentence-transformers (all-MiniLM-L6-v2)
       - all-MiniLM-L6-v2 is 80MB, runs on CPU in ~50ms/sentence, no API cost
       - Consistent with the embedding model used for ChromaDB in ingest.py
         (using the SAME model for both chunking-time similarity and query-time
         retrieval is important — switching models between steps would conflate
         "different chunking" with "different embedding", breaking the isolation
         of chunking as the one variable)
    3. Compute cosine similarity between each consecutive sentence pair
    4. When similarity < threshold, mark a boundary
    5. Merge sentences within each boundary group into one Chunk
    6. If a group exceeds max_tokens, sub-split it using fixed-size logic

Why all-MiniLM-L6-v2 specifically:
    - Official sentence-transformers "getting started" example model
    - Strong performance-per-size tradeoff (384-dim embeddings, 22M params)
    - Downloads once to ~/.cache/huggingface/hub, no re-download per run
    - Zero API cost — important since we run this for 11 docs × Phase 3 iterations

Public API:
    chunk_semantic(doc, threshold=0.4, max_tokens=512) -> list[Chunk]

Parameters:
    threshold:  Cosine similarity threshold below which a sentence boundary
                is placed (default 0.4). Lower = more boundaries (smaller chunks).
                Higher = fewer boundaries (larger chunks that may conflate topics).
                0.4 works well empirically for domain-specific English docs.
    max_tokens: Hard cap on a semantic chunk's size. If a group of sentences
                grouped together exceeds this, it gets sub-split with the fixed-
                size sliding window (same logic as fixed_chunker.py but within
                the group). Default 512 (same as fixed-size for fair comparison).
"""

from __future__ import annotations

import math
from typing import List, Optional

from mcp_server.normalize import NormalizedDocument
from chunking.schema import Chunk, make_chunk
from chunking.utils import split_into_sentences, estimate_tokens


# Embedding model — must match the model used in ingest.py for ChromaDB
# so that chunking-time similarity and query-time retrieval use the same space.
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Lazy-loaded module-level model cache — load once, reuse across all documents.
# Loading takes ~2s on first call; subsequent calls return immediately.
_model = None


def _get_model():
    """Lazy-load the sentence-transformer model (singleton per process)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_EMBEDDING_MODEL)
    return _model


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two numpy vectors."""
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    import numpy as np
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return float(sk_cosine(a, b)[0][0])


def _sentences_to_chunk_text(sentences: List[str]) -> str:
    """Rejoin a group of sentences into chunk text."""
    return " ".join(sentences)


def _split_oversized_group(
    sentences: List[str],
    max_tokens: int,
    doc_id: str,
    source: str,
    title: str,
    doc_type: str,
    path_or_url: Optional[str],
    start_char: int,
    chunk_index_start: int,
) -> List[dict]:
    """
    Sub-split an oversized sentence group using word-count windows.

    This is the fallback for when a semantic boundary group is too large.
    Returns a list of dicts (not Chunk objects) with keys matching make_chunk
    kwargs, so the caller can assign final chunk_index values after all groups
    are processed.
    """
    full_text = _sentences_to_chunk_text(sentences)
    words = full_text.split()
    # Use same words-to-tokens ratio as fixed_chunker
    window_words = max(1, int(max_tokens / 1.33))
    step_words = max(1, int(window_words * 0.875))  # ~12.5% overlap within sub-splits

    sub_chunks = []
    pos = 0
    while pos < len(words):
        end_pos = min(pos + window_words, len(words))
        text = " ".join(words[pos:end_pos])
        approx_start = start_char + len(" ".join(words[:pos])) + (1 if pos > 0 else 0)
        sub_chunks.append({
            "text": text,
            "start_char": approx_start,
            "end_char": approx_start + len(text),
        })
        if end_pos >= len(words):
            break
        pos += step_words
    return sub_chunks


def chunk_semantic(
    doc: NormalizedDocument,
    threshold: float = 0.4,
    max_tokens: int = 512,
) -> List[Chunk]:
    """
    Split a NormalizedDocument into semantic chunks based on embedding similarity.

    Args:
        doc:        A NormalizedDocument (from any MCP connector).
        threshold:  Cosine similarity threshold for boundary placement (default 0.4).
                    Sentences with similarity below this to their neighbor get a
                    chunk boundary placed between them.
        max_tokens: Maximum chunk size in tokens (default 512). Oversized groups
                    are sub-split with fixed-size logic.

    Returns:
        List of Chunk objects. Empty list if doc.content is empty.
    """
    content = doc.content
    if not content or not content.strip():
        return []

    sentences = split_into_sentences(content)
    if not sentences:
        return []

    # Edge case: single sentence
    if len(sentences) == 1:
        return [make_chunk(
            doc_id=doc.doc_id,
            source=doc.source,
            strategy="semantic",
            text=sentences[0],
            chunk_index=0,
            title=doc.title,
            doc_type=doc.metadata.doc_type,
            path_or_url=doc.metadata.path_or_url,
            start_char=0,
            end_char=len(sentences[0]),
            section_header=None,
        )]

    # Embed all sentences in a single batch (efficient)
    model = _get_model()
    embeddings = model.encode(sentences, show_progress_bar=False, batch_size=32)

    # Compute similarity between consecutive sentence pairs
    # and find boundary positions (indices where a new chunk starts)
    boundaries = [0]  # First sentence always starts a new chunk
    for i in range(len(sentences) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        if sim < threshold:
            boundaries.append(i + 1)
    boundaries.append(len(sentences))  # Sentinel

    # Group sentences by their boundary segments
    chunks: List[Chunk] = []
    chunk_index = 0
    # Track char offset through the original content
    char_pos = 0

    for seg_idx in range(len(boundaries) - 1):
        seg_start = boundaries[seg_idx]
        seg_end = boundaries[seg_idx + 1]
        seg_sentences = sentences[seg_start:seg_end]
        seg_text = _sentences_to_chunk_text(seg_sentences)
        seg_tokens = estimate_tokens(seg_text)

        # Approximate start_char: find this segment's text in the content
        # starting from the current char position
        found_pos = content.find(seg_sentences[0], char_pos)
        if found_pos == -1:
            found_pos = char_pos  # fallback if not found (shouldn't happen)
        start_char = found_pos

        if seg_tokens <= max_tokens:
            # Segment fits in one chunk — emit directly
            end_char = start_char + len(seg_text)
            chunks.append(make_chunk(
                doc_id=doc.doc_id,
                source=doc.source,
                strategy="semantic",
                text=seg_text,
                chunk_index=chunk_index,
                title=doc.title,
                doc_type=doc.metadata.doc_type,
                path_or_url=doc.metadata.path_or_url,
                start_char=start_char,
                end_char=end_char,
                section_header=None,  # Semantic strategy does not track headers
            ))
            chunk_index += 1
            char_pos = end_char
        else:
            # Segment is too large — sub-split with fixed-size logic
            sub_parts = _split_oversized_group(
                seg_sentences, max_tokens, doc.doc_id, doc.source,
                doc.title, doc.metadata.doc_type, doc.metadata.path_or_url,
                start_char, chunk_index,
            )
            for part in sub_parts:
                chunks.append(make_chunk(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    strategy="semantic",
                    text=part["text"],
                    chunk_index=chunk_index,
                    title=doc.title,
                    doc_type=doc.metadata.doc_type,
                    path_or_url=doc.metadata.path_or_url,
                    start_char=part["start_char"],
                    end_char=part["end_char"],
                    section_header=None,
                ))
                chunk_index += 1
            char_pos = start_char + len(seg_text)

    return chunks
