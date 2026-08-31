"""
notion_connector.py — Notion API wrapper for AegisRAG MCP server.

Why this file exists:
    Keeps all Notion-specific API logic (auth, pagination, block-to-text
    rendering) isolated from the MCP tool definitions in server.py. The MCP
    tools call exactly two functions from here:
        list_pages()   → list of NormalizedDocument (metadata only, no content)
        fetch_page()   → single NormalizedDocument with full content

Design decisions:
    - Uses `notion-client` SDK (official, handles auth headers + pagination
      transparently). Raw HTTP would need manual header construction and
      paginated cursor loops — not worth it.
    - The Notion block-to-text renderer (`_blocks_to_text`) is intentionally
      minimal: it renders the text content of the most common block types and
      ignores exotic blocks (databases, synced blocks, etc.) with a placeholder.
      The goal for Phase 1 is to get the text into the pipeline faithfully —
      perfect rendering can be a later improvement if needed.
    - `list_pages()` uses `client.search()` (not `databases.query()`) because
      a PAT gives access to pages, not necessarily databases. `search()` with
      filter type="page" returns all pages the integration can see.
    - doc_type inference is keyword-based (simple, no ML) — enough for Phase 3
      metadata filtering.

Environment variables consumed (loaded by caller via python-dotenv):
    NOTION_API_KEY   — Notion Personal Access Token
"""

from __future__ import annotations

import os
import re
from typing import Optional

from notion_client import Client

from .normalize import make_document, NormalizedDocument


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client() -> Client:
    """Instantiate a Notion client from the environment."""
    api_key = os.environ.get("NOTION_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NOTION_API_KEY is not set. "
            "Make sure .env is loaded before calling Notion connector functions."
        )
    return Client(auth=api_key)


def _extract_rich_text(rich_text_array: list) -> str:
    """Flatten a Notion rich_text array (list of text run objects) to a plain string."""
    return "".join(run.get("plain_text", "") for run in rich_text_array)


def _extract_title(page: dict) -> str:
    """
    Pull the page title from a Notion page object.

    Notion stores titles in the 'properties' dict under a key of type 'title'.
    The key name varies per database (could be 'Name', 'Title', or custom) —
    so we iterate properties and find the first one with type == 'title'.
    For top-level pages (not in a database), there is always exactly one title
    property.
    """
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            return _extract_rich_text(prop.get("title", []))
    return "(Untitled)"


def _infer_doc_type(title: str) -> str:
    """
    Simple keyword-based doc_type inference from the page title.

    Returns one of: policy, meeting_notes, spec, checklist, budget, runbook, unknown.
    This maps to the doc_type field in DocumentMetadata — used later for
    metadata filtering in Phase 3 eval queries.
    """
    title_lower = title.lower()
    if any(k in title_lower for k in ("handbook", "policy", "leave", "expense", "security")):
        return "policy"
    if "runbook" in title_lower:
        return "runbook"
    if "meeting" in title_lower or "notes" in title_lower:
        return "meeting_notes"
    if "spec" in title_lower or "specification" in title_lower:
        return "spec"
    if "checklist" in title_lower or "onboarding" in title_lower:
        return "checklist"
    if "budget" in title_lower or "financial" in title_lower:
        return "budget"
    return "unknown"


def _blocks_to_text(blocks: list) -> str:
    """
    Recursively convert a list of Notion block objects to plain text.

    Supported block types (the ones actually used in the Royal Industries corpus):
        paragraph, heading_1/2/3, bulleted_list_item, numbered_list_item,
        to_do, toggle, quote, code, callout, divider

    Unsupported blocks: rendered as "[<type> block — not rendered]\n" so that
    the omission is visible in the output rather than silently lost.

    Why recursive: Notion allows child blocks (nested bullets, toggle content,
    etc.). The `has_children` flag is NOT used here — the caller
    (`fetch_page`) is responsible for pre-fetching all children and passing a
    flat block list. This keeps this function pure/testable.
    """
    lines: list[str] = []

    for block in blocks:
        btype = block.get("type", "")
        bdata = block.get(btype, {})

        if btype in ("paragraph", "quote", "callout"):
            text = _extract_rich_text(bdata.get("rich_text", []))
            lines.append(text)

        elif btype in ("heading_1", "heading_2", "heading_3"):
            text = _extract_rich_text(bdata.get("rich_text", []))
            level = int(btype[-1])          # 1, 2, or 3
            prefix = "#" * level
            lines.append(f"{prefix} {text}")

        elif btype in ("bulleted_list_item", "numbered_list_item", "to_do"):
            text = _extract_rich_text(bdata.get("rich_text", []))
            lines.append(f"- {text}")

        elif btype == "toggle":
            text = _extract_rich_text(bdata.get("rich_text", []))
            lines.append(f"> {text}")

        elif btype == "code":
            text = _extract_rich_text(bdata.get("rich_text", []))
            lang = bdata.get("language", "")
            lines.append(f"```{lang}\n{text}\n```")

        elif btype == "divider":
            lines.append("---")

        elif btype == "child_page":
            # A nested page — we don't recurse into child pages in Phase 1.
            title = bdata.get("title", "(child page)")
            lines.append(f"[Child page: {title}]")

        elif btype in ("image", "video", "file", "pdf", "bookmark"):
            lines.append(f"[{btype} block — not rendered]")

        else:
            lines.append(f"[{btype} block — not rendered]")

        # Recurse into child blocks if pre-fetched (see fetch_page)
        if "children" in block:
            child_text = _blocks_to_text(block["children"])
            if child_text:
                lines.append(child_text)

    return "\n".join(lines)


def _fetch_all_blocks(client: Client, block_id: str) -> list:
    """
    Fetch ALL blocks for a given block_id (page or block), handling pagination.

    Notion paginates block children at 100 per request (default cursor-based).
    This function loops until `has_more` is False, collecting all blocks.
    If a block has children of its own, recurse and attach them as block["children"].
    """
    all_blocks: list[dict] = []
    cursor = None

    while True:
        kwargs: dict = {"block_id": block_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor

        response = client.blocks.children.list(**kwargs)
        results = response.get("results", [])
        all_blocks.extend(results)

        if not response.get("has_more", False):
            break
        cursor = response.get("next_cursor")

    # Recursively fetch children for blocks that have them
    for block in all_blocks:
        if block.get("has_children", False):
            block["children"] = _fetch_all_blocks(client, block["id"])

    return all_blocks


# ---------------------------------------------------------------------------
# Public API (called from server.py)
# ---------------------------------------------------------------------------

def list_pages() -> list[dict]:
    """
    Return metadata for all Notion pages the integration can access.

    Returns a list of dicts (not NormalizedDocuments) because this is a
    metadata-only call — no content fetching needed. Each dict contains:
        page_id, title, last_edited_time, url

    Why `client.search()` and not `databases.query()`:
        A PAT inherits creator access, which is at the page level. `search()`
        with filter type="page" is the correct way to enumerate all accessible
        pages without needing to know database IDs in advance.

    Pagination: `search()` also paginates — handled here with a cursor loop.
    """
    client = _get_client()
    all_pages: list[dict] = []
    cursor = None

    while True:
        kwargs: dict = {
            "filter": {"property": "object", "value": "page"},
            "page_size": 100,
        }
        if cursor:
            kwargs["start_cursor"] = cursor

        response = client.search(**kwargs)
        results = response.get("results", [])
        all_pages.extend(results)

        if not response.get("has_more", False):
            break
        cursor = response.get("next_cursor")

    # Shape results into a lightweight metadata list
    page_list = []
    for page in all_pages:
        page_list.append({
            "page_id": page["id"],
            "title": _extract_title(page),
            "last_edited_time": page.get("last_edited_time"),
            "url": page.get("url"),
        })

    return page_list


def fetch_page(page_id: str) -> NormalizedDocument:
    """
    Fetch the full content of a Notion page and return a NormalizedDocument.

    Steps:
        1. Retrieve page metadata (title, last_edited_time, url)
        2. Retrieve all blocks (with recursive children) via `_fetch_all_blocks`
        3. Convert blocks to plain text via `_blocks_to_text`
        4. Wrap in NormalizedDocument via `make_document`

    The author field is not available from Notion's page API for PAT-based
    integrations without additional workspace-member lookups — left as None.
    """
    client = _get_client()

    # Step 1: page metadata
    page = client.pages.retrieve(page_id=page_id)
    title = _extract_title(page)
    last_edited = page.get("last_edited_time")
    url = page.get("url")
    doc_type = _infer_doc_type(title)

    # Step 2 & 3: blocks → text
    blocks = _fetch_all_blocks(client, page_id)
    content = _blocks_to_text(blocks)

    # Step 4: normalize
    return make_document(
        doc_id=page_id,
        source="notion",
        title=title,
        content=content,
        author=None,        # Not available without extra API call
        modified_date=last_edited,
        doc_type=doc_type,
        path_or_url=url,
    )
