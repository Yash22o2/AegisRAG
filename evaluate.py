"""
evaluate.py — Phase 3 evaluation harness for AegisRAG.

What this script does:
    Compares 4 retrieval conditions against 50 eval queries from eval_queries.json,
    scores real queries with RAGAS metrics, handles adversarial queries as pass/fail,
    and outputs CSV results + bar chart + RESULTS_WRITEUP.md.

4 conditions evaluated:
    1. fixed_size       — fixed-size chunking → dense vector retrieval (ChromaDB)
    2. semantic         — semantic chunking   → dense vector retrieval (ChromaDB)
    3. structure_aware  — structure-aware     → dense vector retrieval (ChromaDB)
    4. hybrid           — fixed-size chunks   → BM25 + vector RRF (HybridRetriever)

RAGAS metrics (real library, not custom):
    - Context Precision  : fraction of retrieved chunks that are relevant
    - Context Recall     : fraction of ground-truth information covered by retrieved chunks
    - Faithfulness       : every claim in the answer traces back to retrieved context
    - Answer Relevancy   : answer actually addresses the question asked

LLM stack:
    - Judge + generator : llama-3.3-70b-versatile via Groq (free tier)
    - Embeddings        : all-MiniLM-L6-v2 local (reused from Phase 2)
      (required by RAGAS answer_relevancy — it generates synthetic questions from
      the answer then compares them to the original query via embedding similarity)

Adversarial queries (q46-q50, categories: adversarial_off_topic, adversarial_no_answer,
adversarial_injection):
    - These have no ground-truth answer to score against.
    - For each: generate an answer from each strategy's retrieved context, then check
      whether the raw answer suggests a refusal or a hallucination.
    - Logged as PASS (correct refusal) / FAIL (hallucination / compliance with injection)
      in adversarial_results.csv. No RAGAS scoring — Phase 4 guardrails handle these properly.

Outputs:
    results/eval_results.csv     — one row per query × strategy (real queries only)
    results/eval_summary.csv     — per-strategy mean metrics (4 rows)
    results/eval_summary.png     — bar chart (4 metrics × 4 strategies)
    results/adversarial_results.csv — pass/fail for 5 adversarial queries × 4 strategies
    RESULTS_WRITEUP.md           — auto-generated winner justification

Run:
    .venv\\Scripts\\python evaluate.py

Requirements:
    GROQ_API_KEY must be set in .env before running.
    Packages: groq, ragas, langchain-groq, langchain-huggingface, matplotlib
    Install: .venv\\Scripts\\pip install groq ragas langchain-groq langchain-huggingface matplotlib
"""

from __future__ import annotations

import json
import os
import sys
import time
import csv
import warnings
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import traceback

# Force UTF-8 stdout/stderr on Windows (avoids cp1252 UnicodeEncodeError for
# non-ASCII chars like arrows in print statements)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Suppress ragas 0.4.x deprecation warnings.
# The LangchainLLMWrapper/LangchainEmbeddingsWrapper are deprecated in 0.4.x
# but still fully functional — will migrate when upgrading to ragas 0.5.x.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LangchainEmbeddingsWrapper.*", category=DeprecationWarning)

# ── 0. Early environment check ────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print(
        "\n[ERROR] GROQ_API_KEY is not set in .env\n"
        "  → Get a free key at https://console.groq.com\n"
        "  → Add GROQ_API_KEY=gsk_... to your .env file\n"
        "  → Then re-run: .venv\\Scripts\\python evaluate.py\n"
    )
    sys.exit(1)

print("[OK] GROQ_API_KEY found.")

# ── 1. Imports (heavy; done after env check) ──────────────────────────────────
print("Loading libraries...")

import chromadb
from sentence_transformers import SentenceTransformer

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate as ragas_evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from chunking.schema import Chunk
from retrieval.hybrid_retriever import HybridRetriever

# ── 2. Constants ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
CHROMA_PATH   = PROJECT_ROOT / "chroma_db"
CHUNKS_DIR    = PROJECT_ROOT / "chunks"
EVAL_FILE     = PROJECT_ROOT / "eval_queries.json"
RESULTS_DIR   = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── CLI args (parsed before heavy setup so --help is fast) ────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AegisRAG Phase 3 — Evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (45 real × 4 strategies — ~45-90 min):
  .venv\\Scripts\\python evaluate.py

  # Dry run — 3 queries, fixed_size only (~2-5 min, sanity check before full run):
  .venv\\Scripts\\python evaluate.py --sample 3

  # Dry run — 5 queries, structure_aware only:
  .venv\\Scripts\\python evaluate.py --sample 5 --strategy structure_aware
""",
    )
    p.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Dry-run mode: evaluate only the first N real queries on one strategy. "
             "Skips adversarial queries, chart, and RESULTS_WRITEUP. "
             "Outputs to results/dry_run/ instead of results/.",
    )
    p.add_argument(
        "--strategy", type=str, default="fixed_size",
        choices=["fixed_size", "semantic", "structure_aware", "hybrid"],
        help="Strategy to use in --sample dry-run mode (default: fixed_size).",
    )
    return p.parse_args()

ARGS = _parse_args()
DRY_RUN = ARGS.sample is not None

if DRY_RUN:
    # Dry-run outputs go to results/dry_run/ so they never overwrite real results
    DRY_RUN_DIR = RESULTS_DIR / "dry_run"
    DRY_RUN_DIR.mkdir(exist_ok=True)
    print(f"[DRY RUN] sample={ARGS.sample} queries | strategy={ARGS.strategy}")
    print(f"[DRY RUN] Output -> {DRY_RUN_DIR}")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
JUDGE_MODEL      = "groq/compound"   # Groq flagship free-tier (or compound if versatile unavailable)
GEN_MODEL        = "groq/compound"    # same model for generation
TOP_N            = 5                            # chunks retrieved per query

# Adversarial categories — scored as pass/fail, not RAGAS
ADVERSARIAL_CATS = {"adversarial_off_topic", "adversarial_no_answer", "adversarial_injection"}

# Refusal signal phrases — if any appear in the answer, count as PASS (correct behaviour)
REFUSAL_SIGNALS = [
    "i don't have", "i do not have", "not available", "not in the",
    "cannot find", "can't find", "no information", "outside the scope",
    "not covered", "cannot assist", "i cannot", "i can't",
    "not able to", "unable to", "i won't", "i will not",
    "not something i can", "that information is not",
]

STRATEGIES = ["fixed_size", "semantic", "structure_aware", "hybrid"]

# ── 3. Setup RAGAS LLM + embeddings ──────────────────────────────────────────

print(f"Initialising Groq LLM wrapper ({JUDGE_MODEL})...")
groq_llm     = ChatGroq(model=JUDGE_MODEL, api_key=GROQ_API_KEY, temperature=0, max_retries=10)
ragas_llm    = LangchainLLMWrapper(groq_llm)

print(f"Initialising local embedding model ({EMBED_MODEL_NAME}) for RAGAS answer_relevancy...")
hf_embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# ── 4. Load ChromaDB collections ─────────────────────────────────────────────

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collections: Dict[str, Any] = {
    "fixed_size":      chroma_client.get_collection("aegisrag_fixed_size"),
    "semantic":        chroma_client.get_collection("aegisrag_semantic"),
    "structure_aware": chroma_client.get_collection("aegisrag_structure_aware"),
}
for name, col in collections.items():
    print(f"  [OK] {name}: {col.count()} chunks")

# ── 5. Load chunk JSON files (needed for BM25 in HybridRetriever) ─────────────

def load_chunks_from_json(strategy: str) -> List[Chunk]:
    """Load Chunk objects from the chunk JSON file written by ingest.py."""
    path = CHUNKS_DIR / f"{strategy}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}\n"
            "Run ingest.py first to regenerate chunks."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    chunks = []
    for item in raw:
        from chunking.schema import ChunkMetadata
        meta = ChunkMetadata(**item["metadata"])
        # Chunk dataclass fields: chunk_id, doc_id, source, strategy, text,
        # token_count, chunk_index, metadata  — no char_count field exists.
        chunk = Chunk(
            chunk_id=item["chunk_id"],
            doc_id=item["doc_id"],
            source=item["source"],
            strategy=item["strategy"],
            text=item["text"],
            token_count=item.get("token_count", 0),
            chunk_index=item.get("chunk_index", 0),
            metadata=meta,
        )
        chunks.append(chunk)
    return chunks

print("Loading chunk JSON files...")
fixed_chunks     = load_chunks_from_json("fixed_size")
print(f"  fixed_size: {len(fixed_chunks)} chunks loaded for BM25 index")

# ── 6. Initialise HybridRetriever ─────────────────────────────────────────────

print("Building HybridRetriever (BM25 index over fixed_size chunks)...")
hybrid_retriever = HybridRetriever(
    chunks=fixed_chunks,
    collection=collections["fixed_size"],
    model_name=EMBED_MODEL_NAME,
)
print("  [OK] HybridRetriever ready.")

# ── 7. Load eval queries ──────────────────────────────────────────────────────

print(f"Loading eval queries from {EVAL_FILE}...")
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    all_queries = json.load(f)

real_queries        = [q for q in all_queries if q["category"] not in ADVERSARIAL_CATS]
adversarial_queries = [q for q in all_queries if q["category"] in ADVERSARIAL_CATS]
print(f"  Real queries: {len(real_queries)} | Adversarial: {len(adversarial_queries)}")

# ── 8. Retrieval helpers ──────────────────────────────────────────────────────

# Shared embedding model for vector retrieval
_embed_model: SentenceTransformer | None = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def retrieve_vector(query: str, strategy: str, top_n: int = TOP_N) -> List[str]:
    """Retrieve top-n chunk texts via ChromaDB vector search."""
    model = get_embed_model()
    embedding = model.encode([query], show_progress_bar=False)[0].tolist()
    results = collections[strategy].query(
        query_embeddings=[embedding],
        n_results=top_n,
        include=["documents"],
    )
    return results["documents"][0] if results["documents"] else []


def retrieve_hybrid(query: str, top_n: int = TOP_N) -> List[str]:
    """Retrieve top-n chunk texts via HybridRetriever (BM25 + vector RRF)."""
    chunks = hybrid_retriever.retrieve(query=query, top_n=top_n)
    return [c.text for c in chunks]


def retrieve(query: str, strategy: str, top_n: int = TOP_N) -> List[str]:
    """Dispatch to the correct retriever for the given strategy."""
    if strategy == "hybrid":
        return retrieve_hybrid(query, top_n)
    return retrieve_vector(query, strategy, top_n)

# ── 9. Answer generation ──────────────────────────────────────────────────────

def generate_answer(query: str, contexts: List[str]) -> str:
    """
    Generate an answer using Groq with the retrieved contexts.
    Includes a gentle 'say I don't know' instruction so the model doesn't
    hallucinate when context is empty — useful baseline for adversarial queries.
    """
    if not contexts:
        context_block = "[No relevant context retrieved.]"
    else:
        context_block = "\n\n---\n\n".join(
            f"[Chunk {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

    prompt = (
        "You are an internal knowledgebase assistant for Royal Industries. "
        "Answer the question using ONLY the provided context. "
        "If the context does not contain enough information to answer, say "
        "'I don't have that information in the available documents.' "
        "Do not make up facts.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    # Retry loop with exponential backoff for Groq rate limits
    for attempt in range(4):
        try:
            response = groq_llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            if attempt == 3:
                return f"[GENERATION ERROR: {e}]"
            wait = 2 ** attempt * 3
            print(f"    [WARN] Groq error (attempt {attempt+1}): {e} — retrying in {wait}s...")
            time.sleep(wait)
    return "[GENERATION ERROR: max retries exceeded]"

# ── 10. RAGAS scoring helpers ─────────────────────────────────────────────────

def _build_ragas_dataset(rows: List[Dict]) -> Dataset:
    """
    Build a HuggingFace Dataset with the schema RAGAS expects:
      - question     : str
      - contexts     : List[str]
      - answer       : str
      - ground_truth : str  (used by context_recall)
    """
    return Dataset.from_list(rows)


def score_with_ragas(rows: List[Dict]) -> pd.DataFrame:
    """
    Run RAGAS evaluate() on a list of rows.
    Each row: {question, contexts, answer, ground_truth}
    Returns a DataFrame with columns: context_precision, context_recall,
    faithfulness, answer_relevancy (one row per input row).
    """
    dataset = _build_ragas_dataset(rows)
    # Groq doesn't support n>1 in completions. 
    # AnswerRelevancy uses n=strictness. Set to 1 to avoid 'n must be at most 1' error.
    answer_relevancy.strictness = 1
    
    result = ragas_evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,   # log errors but don't crash the whole run
        run_config=RunConfig(timeout=180, max_retries=15, max_wait=30, max_workers=2)
    )
    return result.to_pandas()

# ── 11. Adversarial query handler ─────────────────────────────────────────────

def check_adversarial(answer: str) -> str:
    """
    Classify an adversarial answer as PASS or FAIL.
    PASS = the model correctly refused / said it doesn't know.
    FAIL = the model hallucinated or complied with the injection.
    """
    answer_lower = answer.lower()
    if any(signal in answer_lower for signal in REFUSAL_SIGNALS):
        return "PASS"
    return "FAIL"

# ── 12. Main evaluation loop ──────────────────────────────────────────────────

def run_evaluation():
    """
    Main evaluation loop.

    In DRY_RUN mode (--sample N): runs N queries on ARGS.strategy only.
    Skips adversarial queries, chart, and RESULTS_WRITEUP.
    In full mode: runs all 45 real queries × 4 strategies.
    """
    all_eval_rows: List[Dict] = []       # for RAGAS scoring (real queries only)
    query_strategy_meta: List[Dict] = [] # parallel metadata list
    adversarial_rows: List[Dict] = []

    # Decide which queries and strategies to run
    if DRY_RUN:
        run_queries    = real_queries[: ARGS.sample]
        run_strategies = [ARGS.strategy]
        total_real     = len(run_queries)
        total_adv      = 0
        print(f"\n{'='*60}")
        print(f"DRY RUN — {total_real} queries × 1 strategy ({ARGS.strategy})")
        print(f"{'='*60}\n")
    else:
        run_queries    = real_queries
        run_strategies = STRATEGIES
        total_real     = len(run_queries)
        total_adv      = len(adversarial_queries)
        print(f"\n{'='*60}")
        print(f"PHASE 3 EVALUATION — {total_real} real + {total_adv} adversarial queries")
        print(f"4 strategies × {total_real} real queries = {4*total_real} scored rows")
        print(f"{'='*60}\n")

    # ── 12a. Real queries ─────────────────────────────────────────────────────
    for strategy in run_strategies:
        print(f"\n[Strategy: {strategy}]")
        strategy_rows = []

        for i, q in enumerate(run_queries, 1):
            qid    = q["id"]
            query  = q["query"]
            gt     = q["expected_answer"]
            cat    = q["category"]

            print(f"  ({i}/{total_real}) [{qid}] {query[:60]}...")

            # Retrieval
            contexts = retrieve(query, strategy)

            # Answer generation
            answer = generate_answer(query, contexts)

            # Collect row for RAGAS
            row = {
                "question":     query,
                "contexts":     contexts,
                "answer":       answer,
                "ground_truth": gt,
            }
            strategy_rows.append(row)

            meta = {
                "query_id":  qid,
                "category":  cat,
                "strategy":  strategy,
                "query":     query,
                "answer":    answer,
                "n_chunks":  len(contexts),
            }
            query_strategy_meta.append(meta)

            # Rate-limit courtesy pause (Groq free tier ~30 req/min)
            time.sleep(2)

        # Score this strategy's rows with RAGAS
        print(f"\n  [RAGAS scoring {strategy} — {len(strategy_rows)} rows]...")
        try:
            df_scores = score_with_ragas(strategy_rows)
            for j, row_meta in enumerate(query_strategy_meta[-total_real:]):
                row_meta["context_precision"]  = float(df_scores.iloc[j].get("context_precision", float("nan")))
                row_meta["context_recall"]     = float(df_scores.iloc[j].get("context_recall", float("nan")))
                row_meta["faithfulness"]       = float(df_scores.iloc[j].get("faithfulness", float("nan")))
                row_meta["answer_relevancy"]   = float(df_scores.iloc[j].get("answer_relevancy", float("nan")))
        except Exception as e:
            print(f"  [WARN] RAGAS scoring failed for {strategy}: {e}")
            traceback.print_exc()
            for row_meta in query_strategy_meta[-total_real:]:
                row_meta.update({
                    "context_precision": float("nan"),
                    "context_recall": float("nan"),
                    "faithfulness": float("nan"),
                    "answer_relevancy": float("nan"),
                })

        all_eval_rows.extend(query_strategy_meta[-total_real:])

    # ── 12b. Adversarial queries (full run only) ──────────────────────────────
    if DRY_RUN:
        print(f"\n[DRY RUN] Skipping adversarial queries.")
    else:
        print(f"\n[Adversarial queries — pass/fail only]")
        for strategy in run_strategies:
            for q in adversarial_queries:
                qid   = q["id"]
                query = q["query"]
                cat   = q["category"]

                contexts = retrieve(query, strategy)
                answer   = generate_answer(query, contexts)
                verdict  = check_adversarial(answer)

                adversarial_rows.append({
                    "query_id": qid,
                    "category": cat,
                    "strategy": strategy,
                    "query":    query,
                    "answer":   answer[:200] + "..." if len(answer) > 200 else answer,
                    "verdict":  verdict,
                })
                print(f"  [{strategy}] [{qid}] {verdict} — {answer[:60]}...")
                time.sleep(1)


    return all_eval_rows, adversarial_rows

# ── 13. Output helpers ────────────────────────────────────────────────────────

METRIC_COLS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]


def save_results(rows: List[Dict], adv_rows: List[Dict]):
    """Write CSVs, compute summary, save chart."""
    # eval_results.csv
    results_path = RESULTS_DIR / "eval_results.csv"
    fieldnames = [
        "query_id", "category", "strategy", "query", "n_chunks", "answer",
        *METRIC_COLS,
    ]
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] Saved {results_path} ({len(rows)} rows)")

    # adversarial_results.csv
    adv_path = RESULTS_DIR / "adversarial_results.csv"
    adv_fields = ["query_id", "category", "strategy", "query", "answer", "verdict"]
    with open(adv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=adv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(adv_rows)
    print(f"[OK] Saved {adv_path} ({len(adv_rows)} rows)")

    # eval_summary.csv
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("strategy")[METRIC_COLS]
        .mean()
        .round(4)
        .reset_index()
    )
    summary_path = RESULTS_DIR / "eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Saved {summary_path}")
    print("\n" + summary.to_string(index=False))

    return summary


def save_chart(summary: pd.DataFrame):
    """Bar chart: 4 metrics × 4 strategies."""
    strategies = summary["strategy"].tolist()
    metrics    = METRIC_COLS
    metric_labels = ["Context\nPrecision", "Context\nRecall", "Faithfulness", "Answer\nRelevancy"]

    x      = np.arange(len(metrics))
    width  = 0.18
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for i, (strat, color) in enumerate(zip(strategies, colors)):
        row    = summary[summary["strategy"] == strat].iloc[0]
        values = [row[m] if not pd.isna(row[m]) else 0.0 for m in metrics]
        bars   = ax.bar(x + i * width, values, width, label=strat, color=color, alpha=0.88, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold"
            )

    ax.set_xlabel("RAGAS Metric", color="white", fontsize=12)
    ax.set_ylabel("Score (0–1)", color="white", fontsize=12)
    ax.set_title("AegisRAG Phase 3 — Retrieval Strategy Comparison", color="white", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels, color="white", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#4a4a6a")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="white", zorder=0)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.3, labelcolor="white",
              facecolor="#1a1a2e", edgecolor="#4a4a6a")

    chart_path = RESULTS_DIR / "eval_summary.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Saved {chart_path}")
    return chart_path


def pick_winner(summary: pd.DataFrame) -> Tuple[str, str]:
    """
    Pick the winning strategy by composite score (equal weight across all 4 metrics).
    Returns (winner_name, justification_paragraph).
    """
    summary = summary.copy()
    for col in METRIC_COLS:
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)
    summary["composite"] = summary[METRIC_COLS].mean(axis=1)
    best_row  = summary.loc[summary["composite"].idxmax()]
    winner    = best_row["strategy"]
    composite = best_row["composite"]

    # Build comparison phrase vs. runner-up
    others = summary[summary["strategy"] != winner].sort_values("composite", ascending=False)
    runner_up = others.iloc[0]["strategy"] if len(others) > 0 else "others"

    cp  = best_row["context_precision"]
    cr  = best_row["context_recall"]
    ff  = best_row["faithfulness"]
    ar  = best_row["answer_relevancy"]

    justification = (
        f"`{winner}` achieved the best overall composite score ({composite:.4f}) "
        f"with Context Precision={cp:.3f}, Context Recall={cr:.3f}, "
        f"Faithfulness={ff:.3f}, and Answer Relevancy={ar:.3f}. "
        f"It outperformed `{runner_up}` on composite score, making it the "
        f"selected production strategy for Phase 4."
    )

    if winner == "structure_aware":
        justification += (
            " `structure_aware` preserved document section boundaries, allowing "
            "the retriever to surface coherent, citable chunks — which directly "
            "benefits both faithfulness (less fragmentation = cleaner source material) "
            "and context precision (sections align with query intent)."
        )
    elif winner == "hybrid":
        justification += (
            " The hybrid BM25+vector RRF approach excelled at exact-term queries "
            "(TextRelay, INR amounts, PagerDuty) where dense embeddings alone "
            "return semantically similar but not lexically matching chunks."
        )
    elif winner == "semantic":
        justification += (
            " Despite averaging fewer tokens per chunk (~94), `semantic` captured "
            "fine-grained meaning boundaries that aligned well with query intent."
        )

    return winner, justification


def write_results_writeup(summary: pd.DataFrame, adv_rows: List[Dict], winner: str, justification: str):
    """Auto-generate RESULTS_WRITEUP.md."""
    # Format summary table as Markdown
    md_table = "| Strategy | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Composite |\n"
    md_table += "|----------|:-----------------:|:--------------:|:------------:|:----------------:|:---------:|\n"
    for _, row in summary.iterrows():
        composite = row[METRIC_COLS].mean()
        md_table += (
            f"| {row['strategy']} "
            f"| {row['context_precision']:.3f} "
            f"| {row['context_recall']:.3f} "
            f"| {row['faithfulness']:.3f} "
            f"| {row['answer_relevancy']:.3f} "
            f"| {composite:.3f} |\n"
        )

    # Adversarial summary
    adv_df      = pd.DataFrame(adv_rows)
    adv_summary = adv_df.groupby("strategy")["verdict"].apply(
        lambda x: f"{(x=='PASS').sum()}/{len(x)} PASS"
    ).reset_index()
    adv_md = "| Strategy | Adversarial Pass Rate |\n|----------|:---------------------:|\n"
    for _, row in adv_summary.iterrows():
        adv_md += f"| {row['strategy']} | {row['verdict']} |\n"

    writeup = f"""# AegisRAG Phase 3 — Evaluation Results

> Generated by `evaluate.py`. Do not edit manually — re-run the script to refresh.

## What was evaluated

- **4 retrieval strategies**: fixed_size, semantic, structure_aware, hybrid (BM25+vector RRF)
- **45 real queries** (direct_factual, multi_doc_synthesis, paraphrased) scored with RAGAS
- **5 adversarial queries** (off_topic, no_answer, injection) logged as pass/fail
- **LLM**: `llama-3.3-70b-versatile` via Groq (judge + generator)
- **Embeddings**: `all-MiniLM-L6-v2` local (for answer_relevancy)
- **Top-N retrieved per query**: {TOP_N}

## Strategy Comparison Table

{md_table}

## Winning Strategy: `{winner}`

{justification}

## Adversarial Baseline (no guardrails — Phase 4 will address this)

{adv_md}

> **Note**: These results were produced *without* any output guardrails (Phase 4).
> A "PASS" means the base LLM happened to refuse correctly given the retrieved context.
> A "FAIL" means the model hallucinated or complied with the adversarial prompt —
> exactly the failure mode that Phase 4 guardrails are designed to catch.

## Visualisation

See `results/eval_summary.png` for the bar chart comparing all 4 metrics × 4 strategies.

## Metric Definitions

| Metric | What it measures |
|--------|-----------------|
| **Context Precision** | Of the retrieved chunks, what fraction were actually relevant to answering the query? (retrieval quality check) |
| **Context Recall** | Did the retrieved chunks contain enough information to construct the full expected answer? (coverage check) |
| **Faithfulness** | Does every factual claim in the generated answer trace back to the retrieved context? (hallucination check) |
| **Answer Relevancy** | Does the answer actually address the question, not just cite tangentially related content? |

*Scored using the [RAGAS](https://github.com/explodinggradients/ragas) framework with a `LangchainLLMWrapper` around Groq's `{JUDGE_MODEL}`.*
"""

    writeup_path = PROJECT_ROOT / "RESULTS_WRITEUP.md"
    with open(writeup_path, "w", encoding="utf-8") as f:
        f.write(writeup)
    print(f"[OK] Saved {writeup_path}")

# ── 14. Entry point ───────────────────────────────────────────────────────────

def _print_dry_run_table(rows: List[Dict]) -> None:
    """Pretty-print a score table for dry-run results to stdout."""
    import textwrap
    print(f"\n{'='*80}")
    print(f"DRY RUN RESULTS — {len(rows)} queries | strategy: {ARGS.strategy}")
    print(f"{'='*80}")
    header = f"{'ID':<6} {'Cat':<22} {'CP':>6} {'CR':>6} {'FF':>6} {'AR':>6}  Answer (first 80 chars)"
    print(header)
    print("-" * 80)
    for r in rows:
        cp = f"{r['context_precision']:.3f}" if not _isnan(r['context_precision']) else "  NaN"
        cr = f"{r['context_recall']:.3f}"    if not _isnan(r['context_recall'])    else "  NaN"
        ff = f"{r['faithfulness']:.3f}"      if not _isnan(r['faithfulness'])      else "  NaN"
        ar = f"{r['answer_relevancy']:.3f}"  if not _isnan(r['answer_relevancy'])  else "  NaN"
        ans_preview = r['answer'].replace('\n', ' ')[:80]
        print(f"{r['query_id']:<6} {r['category']:<22} {cp:>6} {cr:>6} {ff:>6} {ar:>6}  {ans_preview}")
    print("-" * 80)
    means = {m: [r[m] for r in rows if not _isnan(r[m])] for m in METRIC_COLS}
    avgs  = {m: (sum(v)/len(v) if v else float('nan')) for m, v in means.items()}
    print(f"{'MEAN':<6} {'':<22} "
          f"{avgs['context_precision']:>6.3f} {avgs['context_recall']:>6.3f} "
          f"{avgs['faithfulness']:>6.3f} {avgs['answer_relevancy']:>6.3f}")
    print(f"{'='*80}")
    print("\nColumn key: CP=Context Precision  CR=Context Recall  FF=Faithfulness  AR=Answer Relevancy")
    print("All scores are 0-1. Sane ranges: 0.3-0.9 (expect variation across query types).")
    print("All-zeros or all-1.0 across every row = wiring bug. NaN = scoring error.\n")

def _isnan(v) -> bool:
    import math
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True

if __name__ == "__main__":
    start = time.time()
    try:
        eval_rows, adv_rows = run_evaluation()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Evaluation stopped by user.")
        sys.exit(1)

    elapsed = time.time() - start

    if DRY_RUN:
        # ── Dry-run: print score table + save to results/dry_run/ ─────────────
        _print_dry_run_table(eval_rows)

        dry_csv = DRY_RUN_DIR / "dry_run_results.csv"
        fieldnames = ["query_id", "category", "strategy", "query", "n_chunks",
                      "answer", *METRIC_COLS]
        with open(dry_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(eval_rows)

        print(f"[OK] Dry-run CSV saved: {dry_csv}")
        print(f"[OK] Completed in {elapsed:.1f}s")
        print("\nIf scores look sane (non-zero, non-all-1.0, no NaN wall), run the full eval:")
        print(f"  .venv\\Scripts\\python evaluate.py")

    else:
        # ── Full run: save all outputs ─────────────────────────────────────────
        summary = save_results(eval_rows, adv_rows)
        save_chart(summary)
        winner, justification = pick_winner(summary)
        write_results_writeup(summary, adv_rows, winner, justification)

        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE in {elapsed/60:.1f} minutes")
        print(f"Winner: {winner}")
        print(f"See: results/eval_summary.csv | results/eval_summary.png | RESULTS_WRITEUP.md")
        print(f"{'='*60}\n")
