"""
ingest.py — Phase 2 ingestion pipeline for AegisRAG.

What this script does:
    1. Fetches all 11 documents from the live MCP server (Notion + Drive)
       using the same MCP client pattern as test_mcp_connection.py.
    2. Runs all three chunking strategies on every document:
         - fixed_size  (512 tokens / 64 overlap)
         - semantic    (all-MiniLM-L6-v2, threshold=0.4)
         - structure_aware (Markdown headers, max 512 tokens/section)
    3. For each strategy, embeds all chunks with all-MiniLM-L6-v2 and writes
       them into a dedicated ChromaDB collection:
         - "aegisrag_fixed_size"
         - "aegisrag_semantic"
         - "aegisrag_structure_aware"
    4. Saves all chunks to JSON files in chunks/ directory for inspection
       and Phase 3 use:
         - chunks/fixed_size.json
         - chunks/semantic.json
         - chunks/structure_aware.json
    5. Prints a summary table showing per-strategy chunk counts and timing.

Design decisions:
    - Same embedding model (all-MiniLM-L6-v2) for ALL three strategies' vector
      stores. This is critical for fair comparison: if we used different
      embedding models, we'd be comparing "chunking + embedding" bundles rather
      than chunking in isolation. By keeping embeddings constant, chunking is
      the ONE variable.
    - ChromaDB runs in persistent mode (./chroma_db/ directory) so collections
      survive across runs. On re-run, existing collections are DELETED and
      re-created to ensure a clean state (no stale chunks from previous runs).
    - Notion filtering: we only process pages whose title matches known Royal
      Industries document titles (see ROYAL_INDUSTRIES_TITLES). The 25 Notion
      pages include personal workspace pages — we must exclude those.
    - Drive: all 5 files in the folder are Royal Industries docs, so no filter needed.

Running:
    .venv\\Scripts\\python ingest.py

Expected output:
    Fetching docs via MCP...
    Notion: 6 Royal Industries docs | Drive: 5 docs
    
    Running chunking strategies...
    [fixed_size ] doc 1/11: Engineering Runbook        -> 18 chunks
    ...
    
    Embedding and writing to ChromaDB...
    [fixed_size ] 187 chunks -> aegisrag_fixed_size
    [semantic   ] 143 chunks -> aegisrag_semantic
    [structure_aware] 112 chunks -> aegisrag_structure_aware
    
    Saving chunks to chunks/*.json...
    Done. See chunks/ for inspection.
    
    === SUMMARY ===
    Strategy          | Docs | Chunks | Avg tokens | Time
    fixed_size        |  11  |  187   |    342     |  4.2s
    semantic          |  11  |  143   |    447     |  31.6s
    structure_aware   |  11  |  112   |    398     |  1.1s
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root on sys.path (for `from mcp_server import ...` etc.)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"), override=False)

from mcp_server.normalize import NormalizedDocument
from chunking.fixed_chunker import chunk_fixed
from chunking.semantic import chunk_semantic
from chunking.structure_aware import chunk_structure_aware
from chunking.schema import Chunk


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Royal Industries document titles (exact match against Notion page titles).
# The Notion PAT gives access to 25 pages (including personal workspace pages)
# — we only want these 6. Drive has no such filter needed (all 5 are RI docs).
ROYAL_INDUSTRIES_TITLES = {
    "Engineering Runbook",
    "Expense Policy",
    "Onboarding Guide",
    "IT Security Policy",
    "Leave Policy",
    "Employee Handbook",
}

CHROMA_DIR = os.path.join(_PROJECT_ROOT, "chroma_db")
CHUNKS_DIR = os.path.join(_PROJECT_ROOT, "chunks")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

STRATEGIES = ["fixed_size", "semantic", "structure_aware"]
COLLECTION_NAMES = {
    "fixed_size": "aegisrag_fixed_size",
    "semantic": "aegisrag_semantic",
    "structure_aware": "aegisrag_structure_aware",
}


# ---------------------------------------------------------------------------
# MCP document fetching (reuses same client pattern as test_mcp_connection.py)
# ---------------------------------------------------------------------------

async def fetch_all_documents() -> List[NormalizedDocument]:
    """
    Fetch all 11 Royal Industries documents from the MCP server.

    Returns a list of NormalizedDocument objects (6 Notion + 5 Drive).
    Filters Notion results to only Royal Industries docs by title.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    python_exe = os.path.join(_PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    server_params = StdioServerParameters(
        command=python_exe,
        args=["-m", "mcp_server.server"],
        cwd=_PROJECT_ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
    )

    docs: List[NormalizedDocument] = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ----------------------------------------------------------------
            # Fetch Notion pages
            # ----------------------------------------------------------------
            print("  Fetching Notion pages...", end="", flush=True)
            result = await session.call_tool("notion_list_pages", {})
            pages = [json.loads(item.text) for item in result.content]
            print(f" {len(pages)} total pages found")

            notion_count = 0
            for page in pages:
                title = page.get("title", "")
                if title not in ROYAL_INDUSTRIES_TITLES:
                    continue
                print(f"    Fetching: {title}...", end="", flush=True)
                page_result = await session.call_tool(
                    "notion_fetch_page", {"page_id": page["page_id"]}
                )
                # Single dict result — join all content items
                raw = "".join(item.text for item in page_result.content)
                doc_dict = json.loads(raw)
                doc = _dict_to_normalized_doc(doc_dict)
                docs.append(doc)
                notion_count += 1
                print(f" {len(doc.content)} chars")

            print(f"  Notion: {notion_count} Royal Industries docs fetched")

            # ----------------------------------------------------------------
            # Fetch Drive files
            # ----------------------------------------------------------------
            print("  Fetching Drive files...", end="", flush=True)
            drive_result = await session.call_tool("drive_list_files", {})
            files = [json.loads(item.text) for item in drive_result.content]
            print(f" {len(files)} files found")

            drive_count = 0
            for f in files:
                print(f"    Fetching: {f['name']}...", end="", flush=True)
                file_result = await session.call_tool(
                    "drive_fetch_file", {"file_id": f["file_id"]}
                )
                raw = "".join(item.text for item in file_result.content)
                doc_dict = json.loads(raw)
                doc = _dict_to_normalized_doc(doc_dict)
                docs.append(doc)
                drive_count += 1
                print(f" {len(doc.content)} chars")

            print(f"  Drive: {drive_count} docs fetched")

    print(f"\n  Total documents fetched: {len(docs)}")
    return docs


def _dict_to_normalized_doc(d: dict) -> NormalizedDocument:
    """Reconstruct a NormalizedDocument from a JSON-deserialized dict."""
    from mcp_server.normalize import make_document
    meta = d.get("metadata", {})
    return make_document(
        doc_id=d["doc_id"],
        source=d["source"],
        title=d["title"],
        content=d["content"],
        author=meta.get("author"),
        modified_date=meta.get("modified_date"),
        doc_type=meta.get("doc_type", "unknown"),
        path_or_url=meta.get("path_or_url"),
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def run_chunking(docs: List[NormalizedDocument]) -> Dict[str, List[Chunk]]:
    """
    Run all three chunking strategies on all documents.

    Returns:
        {"fixed_size": [...], "semantic": [...], "structure_aware": [...]}
    """
    all_chunks: Dict[str, List[Chunk]] = {s: [] for s in STRATEGIES}
    timings: Dict[str, float] = {}

    for strategy in STRATEGIES:
        print(f"\n  [{strategy:<17}] chunking {len(docs)} docs...")
        t0 = time.perf_counter()

        for i, doc in enumerate(docs):
            if strategy == "fixed_size":
                chunks = chunk_fixed(doc)
            elif strategy == "semantic":
                chunks = chunk_semantic(doc)
            elif strategy == "structure_aware":
                chunks = chunk_structure_aware(doc)
            else:
                chunks = []

            all_chunks[strategy].extend(chunks)
            print(
                f"    doc {i+1:02d}/{len(docs)}: {doc.title:<35} -> {len(chunks):>3} chunks"
            )

        elapsed = time.perf_counter() - t0
        timings[strategy] = elapsed
        total = len(all_chunks[strategy])
        avg_tokens = (
            sum(c.token_count for c in all_chunks[strategy]) // total if total else 0
        )
        print(
            f"  [{strategy:<17}] done: {total} total chunks | "
            f"avg {avg_tokens} tokens | {elapsed:.1f}s"
        )

    return all_chunks


# ---------------------------------------------------------------------------
# ChromaDB ingestion
# ---------------------------------------------------------------------------

def embed_and_store(
    all_chunks: Dict[str, List[Chunk]],
) -> Dict[str, Any]:
    """
    Embed chunks and write them into ChromaDB collections.

    One collection per strategy. Deletes and re-creates collections on each
    run to ensure clean state.

    Returns:
        dict of {strategy: chromadb.Collection} for use by HybridRetriever.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collections = {}

    for strategy, chunks in all_chunks.items():
        coll_name = COLLECTION_NAMES[strategy]
        print(f"\n  [{strategy:<17}] embedding {len(chunks)} chunks -> {coll_name}...")
        t0 = time.perf_counter()

        # Delete existing collection for clean state
        try:
            client.delete_collection(coll_name)
            print(f"    Deleted existing collection '{coll_name}'")
        except Exception:
            pass  # Collection didn't exist yet

        collection = client.create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine distance
        )

        if not chunks:
            print(f"    No chunks for {strategy} — skipping.")
            collections[strategy] = collection
            continue

        # Batch embed for efficiency
        texts = [c.text for c in chunks]
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embs = model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embs.tolist())

        # Prepare ChromaDB upsert data
        # ChromaDB metadata values must be str, int, float, or bool — no nested dicts.
        # We flatten ChunkMetadata fields into the metadata dict.
        ids = [c.chunk_id for c in chunks]
        metadatas = []
        for c in chunks:
            m = {
                "doc_id": c.doc_id,
                "source": c.source,
                "strategy": c.strategy,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "title": c.metadata.title,
                "doc_type": c.metadata.doc_type,
                "path_or_url": c.metadata.path_or_url or "",
                "start_char": c.metadata.start_char,
                "end_char": c.metadata.end_char,
                "section_header": c.metadata.section_header or "",
            }
            metadatas.append(m)

        # ChromaDB add in batches (limit: 5461 per call)
        chroma_batch = 500
        for i in range(0, len(ids), chroma_batch):
            collection.add(
                ids=ids[i:i + chroma_batch],
                embeddings=all_embeddings[i:i + chroma_batch],
                documents=texts[i:i + chroma_batch],
                metadatas=metadatas[i:i + chroma_batch],
            )

        elapsed = time.perf_counter() - t0
        print(f"    Done: {len(chunks)} chunks in {elapsed:.1f}s")
        collections[strategy] = collection

    return collections


# ---------------------------------------------------------------------------
# Save chunks to JSON
# ---------------------------------------------------------------------------

def save_chunks_json(all_chunks: Dict[str, List[Chunk]]) -> None:
    """Save all chunks to chunks/<strategy>.json for offline inspection."""
    Path(CHUNKS_DIR).mkdir(exist_ok=True)
    for strategy, chunks in all_chunks.items():
        out_path = os.path.join(CHUNKS_DIR, f"{strategy}.json")
        data = [c.to_dict() for c in chunks]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(chunks)} chunks -> {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_chunks: Dict[str, List[Chunk]]) -> None:
    """Print a formatted summary table after ingestion."""
    print("\n" + "=" * 65)
    print("  PHASE 2 INGESTION SUMMARY")
    print("=" * 65)
    print(f"  {'Strategy':<20} {'Chunks':>7} {'Avg tok':>8} {'Min':>6} {'Max':>6}")
    print("  " + "-" * 50)
    for strategy, chunks in all_chunks.items():
        if not chunks:
            print(f"  {strategy:<20} {'0':>7}")
            continue
        toks = [c.token_count for c in chunks]
        avg = sum(toks) // len(toks)
        print(
            f"  {strategy:<20} {len(chunks):>7} {avg:>8} {min(toks):>6} {max(toks):>6}"
        )
    print("=" * 65)
    print(f"\n  ChromaDB collections written to: {CHROMA_DIR}")
    print(f"  Chunk JSON files written to:      {CHUNKS_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n=== AegisRAG Phase 2 — Ingestion Pipeline ===\n")

    # 1. Fetch documents via MCP
    print("[1/4] Fetching documents via MCP server...")
    docs = asyncio.run(fetch_all_documents())

    if not docs:
        print("[ERROR] No documents fetched. Check MCP server and .env config.")
        sys.exit(1)

    # 2. Chunk with all 3 strategies
    print(f"\n[2/4] Running chunking strategies on {len(docs)} documents...")
    all_chunks = run_chunking(docs)

    # 3. Embed and write to ChromaDB
    print("\n[3/4] Embedding chunks and writing to ChromaDB...")
    collections = embed_and_store(all_chunks)

    # 4. Save JSON artifacts
    print("\n[4/4] Saving chunk JSON files...")
    save_chunks_json(all_chunks)

    # Print summary
    print_summary(all_chunks)

    print("\n[DONE] Phase 2 ingestion complete.\n")
    print("Next: Run Phase 3 evaluation with eval_queries.json")


if __name__ == "__main__":
    main()
