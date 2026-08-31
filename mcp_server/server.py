"""
server.py — FastMCP server for AegisRAG.

Why this file exists:
    This is the entry point of the MCP server. It defines the four tools that
    any MCP-compatible client (Claude Desktop, LangGraph agent, test script)
    can discover and call without knowing anything about Notion or Drive
    directly — that's the value of the MCP abstraction layer.

Tools exposed:
    notion_list_pages   → list page metadata from Notion workspace
    notion_fetch_page   → fetch full content of one Notion page by page_id
    drive_list_files    → list file metadata from the scoped Drive folder
    drive_fetch_file    → fetch full content of one Drive file by file_id

Design decisions:
    - FastMCP from the official `mcp` Python SDK is used instead of the lower-
      level Server class because it dramatically reduces boilerplate: just
      @mcp.tool() decorators on regular functions. The raw Server API requires
      manually registering handlers, building response objects, etc.
    - All four tool functions are thin wrappers — they load .env, call a
      connector function, and return JSON-serializable output. All real logic
      lives in the connector modules (independently testable without MCP).
    - Return type is always a plain dict (JSON-serializable) — MCP serializes
      tool return values; the connector's .to_dict() converts dataclasses.
    - Transport: stdio (run as a subprocess, communicate via stdin/stdout).
      This is the simplest deployment for local/portfolio use — no HTTP server
      needed, no port to manage, works directly with `mcp run server.py` or
      any subprocess-based MCP client.
    - .env is loaded at module import time (not per-call) to avoid repeated
      file reads. The dotenv library is idempotent: calling load_dotenv()
      multiple times doesn't cause issues.

Running this server:
    Option 1 (direct):  .venv/Scripts/python mcp_server/server.py
    Option 2 (MCP CLI): .venv/Scripts/mcp run mcp_server/server.py
    Option 3 (test):    Use test_mcp_connection.py (calls via stdio client)
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the project root (parent of mcp_server/) is on sys.path.
# This is needed when server.py is launched as a subprocess from the test script
# or directly as `python mcp_server/server.py` — in those cases, Python adds
# mcp_server/ to sys.path (the script's directory), not the project root, so
# `from mcp_server import ...` would fail without this.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

# Load .env before importing connectors so NOTION_API_KEY, GOOGLE_CREDENTIALS_PATH,
# and DRIVE_FOLDER_ID are available when the connector modules are first imported.
# override=False means existing environment variables (e.g., set in shell/CI) win.
# dotenv searches for .env starting from the script's directory and walking up;
# since we set CWD to project root in the test, this resolves correctly.
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"), override=False)

# Now import connectors using relative imports (same package, avoids sys.path issues)
from . import notion_connector, drive_connector


# ---------------------------------------------------------------------------
# MCPServer instantiation (mcp v2 — FastMCP was renamed to MCPServer in 2.x)
# ---------------------------------------------------------------------------

mcp = MCPServer(
    name="AegisRAG",
    instructions=(
        "This MCP server exposes Royal Industries' internal knowledge base "
        "via four tools: two for Notion (list pages, fetch page content) and "
        "two for Google Drive (list files, fetch file content). "
        "All tools return documents in a normalized schema: "
        "{doc_id, source, title, content, metadata}."
    ),
)


# ---------------------------------------------------------------------------
# Tool: notion_list_pages
# ---------------------------------------------------------------------------

@mcp.tool()
def notion_list_pages() -> list[dict]:
    """
    List all Notion pages accessible to the AegisRAG integration.

    Returns a list of page metadata objects. Each object contains:
      - page_id (str): Unique Notion page identifier. Pass this to
        notion_fetch_page() to retrieve the full content.
      - title (str): Human-readable page title.
      - last_edited_time (str | null): ISO 8601 timestamp of last edit.
      - url (str | null): Direct Notion URL for the page.

    Does NOT return page content — use notion_fetch_page(page_id) for that.
    """
    return notion_connector.list_pages()


# ---------------------------------------------------------------------------
# Tool: notion_fetch_page
# ---------------------------------------------------------------------------

@mcp.tool()
def notion_fetch_page(page_id: str) -> dict:
    """
    Fetch the full text content of a Notion page.

    Args:
      page_id: The Notion page ID (from notion_list_pages results).
               Accepts both the raw UUID format (32 hex chars) and the
               dashed format (8-4-4-4-12) — Notion's API handles both.

    Returns a normalized document object:
      {
        "doc_id": str,
        "source": "notion",
        "title": str,
        "content": str,          # Full page text, block-rendered to markdown-ish plain text
        "metadata": {
          "author": null,        # Not available via PAT without extra API call
          "modified_date": str,  # ISO 8601
          "doc_type": str,       # Inferred: "policy" | "runbook" | "checklist" | etc.
          "path_or_url": str     # Direct Notion URL
        }
      }
    """
    doc = notion_connector.fetch_page(page_id)
    return doc.to_dict()


# ---------------------------------------------------------------------------
# Tool: drive_list_files
# ---------------------------------------------------------------------------

@mcp.tool()
def drive_list_files() -> list[dict]:
    """
    List all files in the AegisRAG Google Drive folder.

    Returns a list of file metadata objects. Each object contains:
      - file_id (str): Unique Drive file ID. Pass this to drive_fetch_file()
        to retrieve the full content.
      - name (str): File name as it appears in Drive.
      - mime_type (str): MIME type (e.g., "application/vnd.google-apps.document").
      - modified_time (str | null): ISO 8601 timestamp of last modification.
      - web_view_link (str | null): URL to view the file in the browser.
      - author (str | null): Display name of the file owner.

    Scoped to folder ID configured in DRIVE_FOLDER_ID environment variable.
    Does NOT return file content — use drive_fetch_file(file_id) for that.
    """
    return drive_connector.list_files()


# ---------------------------------------------------------------------------
# Tool: drive_fetch_file
# ---------------------------------------------------------------------------

@mcp.tool()
def drive_fetch_file(file_id: str) -> dict:
    """
    Fetch the full text content of a Google Drive file.

    Handles the following file types automatically:
      - Google Docs      → exported as plain text
      - Google Sheets    → exported as CSV (Sheet1) + plain text (all sheets)
      - Google Slides    → exported as plain text
      - PDFs             → text extracted via pypdf
      - Other text files → best-effort text extraction

    Args:
      file_id: The Drive file ID (from drive_list_files results).

    Returns a normalized document object:
      {
        "doc_id": str,
        "source": "drive",
        "title": str,
        "content": str,          # Extracted text content
        "metadata": {
          "author": str | null,  # File owner display name
          "modified_date": str,  # ISO 8601
          "doc_type": str,       # Inferred: "meeting_notes" | "spec" | "budget" | etc.
          "path_or_url": str     # Web view link
        }
      }

    Note: On first call, a browser window will open for Google OAuth consent.
    After authorizing, a token.json is cached locally for subsequent calls.
    """
    doc = drive_connector.fetch_file(file_id)
    return doc.to_dict()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
