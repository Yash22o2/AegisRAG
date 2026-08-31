# AegisRAG

> **An MCP-based RAG ingestion pipeline with a rigorous chunking-strategy evaluation harness** — built for Royal Industries' internal knowledgebase, demonstrating real-world AI engineering over a live multi-source corpus.

---

## What This Is

AegisRAG is a portfolio-grade AI engineering project that goes well beyond "upload PDFs to a vector DB." It is structured in four phases:

| Phase | What | Status |
|-------|------|--------|
| **0 — Corpus** | 11 synthetic Royal Industries documents across Notion + Drive | ✅ Complete |
| **1 — MCP Server** | Live Notion + Google Drive connectors behind a unified MCP tool interface | ✅ Complete |
| **2 — Chunking** | Four distinct chunking strategies implemented and compared | 🔄 In progress |
| **3 — Eval Harness** | RAGAS-scored comparison across 50 curated queries with ground truth | 🔜 Planned |
| **4 — RAG Pipeline** | Citation-backed generation with output guardrails | 🔜 Planned |

The project's defining feature is **comparative rigor** — rather than assuming one chunking strategy works best, it builds a harness that surfaces real metric differences (Context Precision, Context Recall, Answer Faithfulness) across strategies on a carefully designed corpus.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP Clients                          │
│        (Claude Desktop · LangGraph agent · test script)     │
└───────────────────────┬─────────────────────────────────────┘
                        │  stdio transport
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   mcp_server/server.py                      │
│                                                             │
│  @tool  notion_list_pages    @tool  notion_fetch_page       │
│  @tool  drive_list_files     @tool  drive_fetch_file        │
└──────────────┬────────────────────────┬────────────────────┘
               │                        │
               ▼                        ▼
┌──────────────────────┐   ┌───────────────────────┐
│  notion_connector.py │   │  drive_connector.py   │
│                      │   │                       │
│  · PAT auth          │   │  · OAuth 2.0 + cache  │
│  · Paginated search  │   │  · MIME-type routing  │
│  · Block → text      │   │  · Sheets dual-export │
│    (recursive)       │   │  · PDF text extract   │
└──────────┬───────────┘   └──────────┬────────────┘
           │                          │
           └────────────┬─────────────┘
                        ▼
             ┌─────────────────────┐
             │    normalize.py     │
             │  NormalizedDocument │
             │  {doc_id, source,   │
             │   title, content,   │
             │   metadata}         │
             └─────────────────────┘
                        │
                        ▼  (Phase 2 →)
             ┌─────────────────────┐
             │  chunking/          │
             │  · fixed_size.py    │
             │  · semantic.py      │
             │  · structural.py    │
             │  · agentic.py       │
             └─────────────────────┘
```

### Why MCP?

The Model Context Protocol (MCP) defines a standard tool interface that any compatible client can call without custom per-source integration. By building Notion and Drive connectors behind MCP tools rather than hardcoding them into the RAG pipeline, this project demonstrates:

- **Source agnosticism**: the chunking and eval pipeline never knows or cares whether a document came from Notion or Drive
- **Reusability**: any MCP-compatible agent (Claude Desktop, LangGraph, custom) can call these tools without modification
- **Clean separation**: the ingestion layer and the RAG layer are independently testable and swappable

---

## The Corpus — Royal Industries

All documents are synthetic but written to realistic depth (~1,500–2,500 words each), intentionally cross-referencing each other to enable multi-document synthesis queries in the eval set.

**Notion (6 documents — structured policy/handbook content):**
- `hr_handbook.md` — 9-section employee handbook
- `leave_policy.md` — 6 leave types, accrual rules, edge cases
- `it_security_policy.md` — passwords, MFA, data classification tiers, BYOD, AI tool policy
- `onboarding_guide.md` — pre-boarding through 90-day milestones, system access routing
- `expense_policy.md` — travel, per diem, approval thresholds, corporate cards
- `engineering_runbook.md` — Aurora platform architecture, 4 incident playbooks, on-call rotation

**Google Drive (5 documents — intentionally messier, mixed formats):**
- `meeting_notes_q3_planning` — informal, scattered action items (stress-tests chunking)
- `product_spec_aurora` — draft spec with open questions, cross-references
- `onboarding_checklist` — checklist format, technical onboarding supplement
- `vendor_contract_summary` — SMS provider (TextRelay) contract summary
- `q3_budget_summary` — Google Sheet with structured table + prose narrative

**Why the deliberate format variety?** Fixed-size chunking handles clean policy docs well but struggles with meeting notes and tables. Semantic/structure-aware chunking has the opposite profile. The corpus is designed so all four chunking strategies have different performance profiles — which is what makes the evaluation harness's output meaningful.

---

## Phase 1 — MCP Server (Complete)

### What the server exposes

```python
# List all accessible Notion pages (metadata only)
notion_list_pages() -> list[{page_id, title, last_edited_time, url}]

# Fetch full text content of a Notion page
notion_fetch_page(page_id: str) -> NormalizedDocument

# List all files in the scoped Drive folder
drive_list_files() -> list[{file_id, name, mime_type, modified_time, ...}]

# Fetch full text content of a Drive file (auto-routes by MIME type)
drive_fetch_file(file_id: str) -> NormalizedDocument
```

### Normalized document schema

Every tool returns data in this shape — source-agnostic, JSON-serializable:

```json
{
  "doc_id": "3c83fc80-9e11-8046-8f51-c6c255ed4d66",
  "source": "notion",
  "title": "Engineering Runbook",
  "content": "# Royal Industries — Engineering Runbook: Aurora Sensor Platform...",
  "metadata": {
    "author": null,
    "modified_date": "2026-08-26T12:48:00.000Z",
    "doc_type": "runbook",
    "path_or_url": "https://app.notion.com/p/Engineering-Runbook-..."
  }
}
```

### Technical implementation notes

**Notion connector:**
- Uses the official `notion-client` SDK with a Personal Access Token (PAT)
- PAT inherits creator's workspace access — no per-page sharing required
- `client.search(filter={"type": "page"})` with cursor-based pagination to enumerate all pages
- Recursive block fetching (`_fetch_all_blocks`) handles Notion's nested block model
- Block-to-text renderer covers all common block types (headings, lists, callouts, code, toggles, dividers)

**Drive connector:**
- OAuth 2.0 Desktop flow with `token.json` caching — one-time browser auth, silent on subsequent runs
- Scoped to a single Drive folder (`DRIVE_FOLDER_ID`) — never touches files outside the corpus
- MIME-type-aware content extraction:
  - Google Docs → `text/plain` export
  - Google Sheets → dual pass: `text/csv` (Sheet1 table) + `text/plain` (all sheets) — captures both structured table data and prose narrative
  - PDFs → binary download + `pypdf` text extraction
  - Unknown types → best-effort text download with fallback

**mcp v2 note (for contributors):** This project uses `mcp==2.1.1` which is the v2 SDK. `FastMCP` was renamed to `MCPServer` at `mcp.server.mcpserver`. Always run the server as a module (`python -m mcp_server.server`), not as a script — relative imports require module mode.

---

## Setup

### Prerequisites

- Python 3.10+
- A Notion workspace with a Personal Access Token (PAT) — [create one here](https://www.notion.so/profile/integrations)
- A Google Cloud project with Drive API enabled and OAuth 2.0 Desktop credentials downloaded as `credentials.json`

### Installation

```bash
# Clone the repo
git clone https://github.com/Yash22o2/AegisRAG.git
cd AegisRAG

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root (see `.env.example`):

```env
NOTION_API_KEY=your_notion_pat_here
GOOGLE_CREDENTIALS_PATH=credentials.json
DRIVE_FOLDER_ID=your_drive_folder_id_here
```

Place your `credentials.json` in the project root.

### First-time Google Drive authorization

```bash
python authorize_drive.py
```

This opens a browser for Google OAuth consent. After you click Allow, `token.json` is cached and no browser is needed for subsequent runs.

### Run the end-to-end test

```bash
python test_mcp_connection.py
```

Expected output:

```
>> Starting AegisRAG MCP server as subprocess...
   Server initialized [OK]

TEST 1: Tool Discovery (list_tools)
  [OK]  All 4 tools discovered: [drive_fetch_file, drive_list_files, notion_fetch_page, notion_list_pages]

TEST 2: notion_list_pages
  [OK]  N pages returned from Notion
  [OK]  Page schema valid. First page: 'Engineering Runbook'

TEST 3: notion_fetch_page
  [OK]  Content fetched, length: 7210 chars
  [OK]  Document schema valid

TEST 4: drive_list_files
  [OK]  5 files returned from Drive
  [OK]  File schema valid. First file: 'Q3 Budget Summary'

TEST 5: drive_fetch_file
  [OK]  Content fetched, length: 576 chars
  [OK]  Document schema valid
```

---

## Project Structure

```
AegisRAG/
├── mcp_server/
│   ├── __init__.py
│   ├── server.py              # MCPServer entry point, 4 tool definitions
│   ├── notion_connector.py    # Notion API wrapper
│   ├── drive_connector.py     # Google Drive API wrapper
│   └── normalize.py           # NormalizedDocument schema + factory
├── authorize_drive.py         # One-time Drive OAuth helper
├── test_mcp_connection.py     # End-to-end MCP client test / demo
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
└── README.md
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `mcp` | 2.1.1 | MCP Python SDK (MCPServer, stdio transport, client) |
| `notion-client` | 3.1.0 | Official Notion API SDK |
| `google-api-python-client` | 2.199.0 | Google Drive API |
| `google-auth-oauthlib` | 1.4.1 | OAuth 2.0 Desktop flow |
| `google-auth-httplib2` | 0.4.2 | HTTP transport for Google auth |
| `python-dotenv` | 1.2.3 | `.env` loading |
| `pypdf` | 6.16.2 | PDF text extraction |

---

## Roadmap

- [x] **Phase 0** — Synthetic corpus (11 docs, ~13,000 words, cross-referenced)
- [x] **Phase 1** — MCP server with Notion + Drive connectors, normalized schema
- [ ] **Phase 2** — Four chunking strategies: fixed-size · semantic · structure-aware · agentic
- [ ] **Phase 3** — Evaluation harness: 50-query ground-truth set, RAGAS metrics, comparison table
- [ ] **Phase 4** — Full RAG pipeline: retrieval → generation with citation formatting + output guardrails

---

## Why This Project

Most RAG demos:
1. Use a single PDF or pre-chunked dataset
2. Apply one chunking strategy without justification
3. Evaluate qualitatively ("it seemed to work")

AegisRAG is built to answer the question: **for a realistic, multi-source enterprise knowledgebase, which chunking strategy actually performs better, and by how much?** The answer is in the numbers — Context Recall scores, not vibes.

---

## License

MIT
