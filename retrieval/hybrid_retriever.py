"""
hybrid_retriever.py — Hybrid BM25 + vector search retriever for AegisRAG.

Why this strategy exists (and why it's retrieval-axis, not chunking-axis):
    The three chunking strategies all feed their chunks into a vector store
    (ChromaDB) and are compared on the same retrieval axis: dense vector search
    with the same embedding model (all-MiniLM-L6-v2). This isolates chunking
    as the one variable being evaluated.

    The hybrid retriever is a FOURTH comparison point, but on a different axis:
    it uses fixed-size chunks (the simplest chunking, so chunking isn't the
    variable) and asks: "does adding keyword search (BM25) on top of vector
    search improve retrieval?" This is a genuine and important question for
    our corpus because:

    - Dense embeddings excel at semantic matching ("data breach" ↔ "security
      incident") but can miss exact string matches.
    - Our corpus contains many exact-match critical terms: "TextRelay",
      "INR 2,40,00,000", "Aurora", "PagerDuty", "DRIVE_FOLDER_ID". A query
      for "TextRelay contract" should DEFINITELY retrieve the vendor contract
      chunk — BM25 catches this even if the embedding similarity is modest.
    - Reciprocal Rank Fusion (RRF) is the standard, parameter-light fusion
      method. It avoids score normalization (BM25 and cosine live on different
      scales) and is robust to outliers.

Algorithm — Reciprocal Rank Fusion (RRF):
    1. BM25 rank: rank all chunks by BM25 score against the query
    2. Vector rank: rank all chunks by cosine similarity against the query
       (via ChromaDB query)
    3. For each chunk, compute RRF score = 1/(k + rank_bm25) + 1/(k + rank_vector)
       where k=60 is the standard RRF constant (dampens the impact of very high
       rank differences)
    4. Return top-n chunks sorted by RRF score descending

Implementation details:
    - BM25: rank_bm25 library (BM25Okapi) — pure Python, no external index
    - Vector: ChromaDB collection queried by the same model as ingest.py
    - The HybridRetriever is initialized with the fixed_size ChromaDB collection
      and the full list of fixed-size Chunk objects (for BM25 indexing)
    - At query time: runs both searches, fuses with RRF, returns top-n Chunks

Usage (from Phase 3 eval or Phase 4 RAG pipeline):
    retriever = HybridRetriever(chunks=fixed_chunks, collection=fixed_collection)
    results = retriever.retrieve(query="What is the on-call rotation?", top_n=5)
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional

from chunking.schema import Chunk


# RRF constant — standard value, trades off between top-rank dominance and
# rank smoothing. k=60 is the original Cormack et al. recommendation.
_RRF_K = 60


class HybridRetriever:
    """
    Hybrid BM25 + ChromaDB vector retriever using Reciprocal Rank Fusion.

    Initialize once per session with the chunks and ChromaDB collection for
    a specific chunking strategy (typically fixed_size, since that's what
    the hybrid comparison is built on top of).

    Args:
        chunks:     List of Chunk objects that were ingested into `collection`.
                    Used to build the in-memory BM25 index and to map
                    retrieved chunk_ids back to Chunk objects.
        collection: A chromadb.Collection already populated with these chunks.
        model_name: Embedding model name (must match what was used in ingest.py).
    """

    def __init__(
        self,
        chunks: List[Chunk],
        collection,  # chromadb.Collection — typed loosely to avoid import at class definition time
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.chunks = chunks
        self.collection = collection
        self.model_name = model_name

        # Build a dict for fast chunk_id → Chunk lookup
        self._chunk_map: Dict[str, Chunk] = {c.chunk_id: c for c in chunks}

        # Build BM25 index over tokenized chunk texts
        # BM25Okapi expects a list of token lists (word-tokenized)
        self._corpus_tokens = [c.text.lower().split() for c in chunks]
        self._chunk_ids_ordered = [c.chunk_id for c in chunks]

        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._corpus_tokens)

        # Lazy-load embedding model (shared with semantic.py via module-level cache)
        self._embed_model = None

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.model_name)
        return self._embed_model

    def _bm25_ranks(self, query: str) -> Dict[str, int]:
        """
        Return a dict of {chunk_id: 0-based rank} for BM25 results.

        BM25 ranks from highest score (rank 0) to lowest.
        """
        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)
        # Sort chunk_ids by score descending → rank 0 = best
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )
        return {self._chunk_ids_ordered[i]: rank for rank, i in enumerate(ranked)}

    def _vector_ranks(self, query: str, top_n: int) -> Dict[str, int]:
        """
        Return a dict of {chunk_id: 0-based rank} for ChromaDB vector results.

        Only retrieves up to max(top_n * 3, 50) results to keep RRF meaningful —
        chunks not in this result set get a rank of len(chunks) (worst possible).
        """
        model = self._get_embed_model()
        query_embedding = model.encode([query], show_progress_bar=False)[0].tolist()

        n_results = min(max(top_n * 3, 50), len(self.chunks))
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"],
        )

        # results["ids"][0] is a list of chunk_ids in rank order (best first)
        ranked_ids = results["ids"][0] if results["ids"] else []
        ranks = {chunk_id: rank for rank, chunk_id in enumerate(ranked_ids)}

        # All chunks NOT in the result set get a worst-case rank
        worst_rank = len(self.chunks)
        for chunk_id in self._chunk_ids_ordered:
            if chunk_id not in ranks:
                ranks[chunk_id] = worst_rank

        return ranks

    def _rrf_score(self, bm25_rank: int, vector_rank: int) -> float:
        """Compute the RRF score for a single chunk."""
        return 1.0 / (_RRF_K + bm25_rank) + 1.0 / (_RRF_K + vector_rank)

    def retrieve(self, query: str, top_n: int = 5) -> List[Chunk]:
        """
        Retrieve the top-n chunks for a query using Hybrid BM25 + vector RRF.

        Args:
            query:  The user's query string.
            top_n:  Number of chunks to return (default 5).

        Returns:
            List of Chunk objects, sorted by RRF score descending (best first).
        """
        if not self.chunks:
            return []

        bm25_ranks = self._bm25_ranks(query)
        vector_ranks = self._vector_ranks(query, top_n)

        # Compute RRF score for every chunk
        scored: List[Tuple[float, str]] = []
        for chunk_id in self._chunk_ids_ordered:
            bm25_r = bm25_ranks.get(chunk_id, len(self.chunks))
            vector_r = vector_ranks.get(chunk_id, len(self.chunks))
            score = self._rrf_score(bm25_r, vector_r)
            scored.append((score, chunk_id))

        # Sort by score descending, take top_n
        scored.sort(key=lambda x: x[0], reverse=True)
        top_ids = [chunk_id for _, chunk_id in scored[:top_n]]

        return [self._chunk_map[cid] for cid in top_ids if cid in self._chunk_map]

    def retrieve_with_scores(self, query: str, top_n: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Same as retrieve() but also returns the RRF score for each result.

        Useful for Phase 3 evaluation analysis and debugging.

        Returns:
            List of (Chunk, rrf_score) tuples, sorted by score descending.
        """
        if not self.chunks:
            return []

        bm25_ranks = self._bm25_ranks(query)
        vector_ranks = self._vector_ranks(query, top_n)

        scored: List[Tuple[float, str]] = []
        for chunk_id in self._chunk_ids_ordered:
            bm25_r = bm25_ranks.get(chunk_id, len(self.chunks))
            vector_r = vector_ranks.get(chunk_id, len(self.chunks))
            score = self._rrf_score(bm25_r, vector_r)
            scored.append((score, chunk_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored[:top_n]

        return [
            (self._chunk_map[cid], score)
            for score, cid in top_scored
            if cid in self._chunk_map
        ]
