"""
test_mcp_connection.py -- End-to-end MCP client test for AegisRAG.

Purpose:
    Verify that all four MCP tools are discoverable and return real content
    from the live Notion workspace and Google Drive folder. This script is both
    a functional test and a portfolio artifact — it demonstrates that the MCP
    layer actually works end-to-end against real external APIs, which is more
    convincing than describing it.

What it tests (in order):
    1. list_tools()             — all four tools discoverable by an MCP client
    2. notion_list_pages()      — returns ≥1 page with expected fields
    3. notion_fetch_page()      — fetches the first listed page, content is non-empty
    4. drive_list_files()       — returns ≥1 file with expected fields
    5. drive_fetch_file()       — fetches the first listed file, content is non-empty

    For Notion and Drive fetch tests, the script also prints the first 300 chars
    of content so you can visually verify it's real document text (not garbage).

Running:
    .venv\\Scripts\\python test_mcp_connection.py

    On first run, step 4/5 will open a browser tab for Google OAuth consent.
    After authorizing, token.json is cached and subsequent runs skip the browser.

Design decisions:
    - Uses `mcp` Python SDK's `stdio_client` + `ClientSession` to communicate
      with the server as a real MCP client would. This tests the full protocol
      stack (tool discovery, call serialization, response deserialization) —
      not just the Python functions directly.
    - The server is launched as a subprocess (`StdioServerParameters` with the
      path to server.py) so the test is completely self-contained.
    - Assertions use simple `assert` statements (not pytest) to keep this
      runnable as a standalone script without any test framework dependency.
    - Results are printed to stdout in a structured format — useful both for
      debugging and for showing in a portfolio/interview.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=False)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_pass(msg: str) -> None:
    print(f"  [OK]  {msg}")


def print_fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def print_info(label: str, value) -> None:
    if isinstance(value, (dict, list)):
        print(f"  {label}:\n    {json.dumps(value, indent=2, default=str)[:400]}")
    else:
        print(f"  {label}: {str(value)[:400]}")


def parse_list_result(result) -> list:
    """
    Parse a list tool result from mcp v2.

    In mcp 2.x, when a tool returns a list[dict], each element becomes a
    SEPARATE TextContent item in result.content. So we collect all content
    items and parse each one individually.

    Falls back to parsing content[0].text as a JSON array in case behavior
    changes in future mcp versions.
    """
    if not result.content:
        return []

    # Try: collect all TextContent items, each is one serialized element
    items = []
    for c in result.content:
        if hasattr(c, 'text') and c.text:
            try:
                parsed = json.loads(c.text)
                if isinstance(parsed, list):
                    # content[0].text is already a full JSON array -- mcp v1 behavior
                    return parsed
                items.append(parsed)
            except json.JSONDecodeError:
                pass  # Skip non-JSON content items
    return items


def parse_dict_result(result) -> dict:
    """
    Parse a single-dict tool result from mcp v2.

    When a tool returns a single dict, it may come back as one TextContent
    item (with the full JSON), or spread across multiple TextContent items
    (one per field). Collect all text and try to parse.
    """
    if not result.content:
        return {}

    # Concatenate all text content and try to parse as JSON
    full_text = "".join(c.text for c in result.content if hasattr(c, 'text') and c.text)
    try:
        parsed = json.loads(full_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: try just the first item
    try:
        return json.loads(result.content[0].text)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Async test functions
# ---------------------------------------------------------------------------

async def run_tests():
    """
    Connect to the AegisRAG MCP server and run all four tool tests.

    The MCP client communicates with the server via stdio — the server is
    launched as a subprocess.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    project_root = os.path.dirname(os.path.abspath(__file__))
    python_exe = os.path.join(project_root, ".venv", "Scripts", "python.exe")

    # On non-Windows, python.exe is just python3
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    server_params = StdioServerParameters(
        command=python_exe,
        args=["-m", "mcp_server.server"],  # Run as module so relative imports work
        env=dict(os.environ),   # Pass current env (includes loaded .env vars)
        cwd=project_root,       # Set CWD to project root so dotenv finds .env
    )

    print("\n>> Starting AegisRAG MCP server as subprocess...")
    print(f"   Python: {python_exe}")
    print(f"   CWD:    {project_root}")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("   Server initialized [OK]")

            # ----------------------------------------------------------
            # Test 1: Tool discovery
            # ----------------------------------------------------------
            print_header("TEST 1: Tool Discovery (list_tools)")

            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print_info("Discovered tools", tool_names)

            expected_tools = {
                "notion_list_pages",
                "notion_fetch_page",
                "drive_list_files",
                "drive_fetch_file",
            }
            missing = expected_tools - set(tool_names)
            if missing:
                print_fail(f"Missing tools: {missing}")
                return
            print_pass(f"All 4 tools discovered: {sorted(tool_names)}")

            # ----------------------------------------------------------
            # Test 2: notion_list_pages
            # ----------------------------------------------------------
            print_header("TEST 2: notion_list_pages")

            pages_result = await session.call_tool("notion_list_pages", {})
            pages = parse_list_result(pages_result)

            print_info("Page count", len(pages))
            if not pages:
                print_fail("No pages returned — check NOTION_API_KEY and workspace access")
                return
            print_pass(f"{len(pages)} pages returned from Notion")

            # Validate schema of first page
            first_page = pages[0]
            required_fields = {"page_id", "title", "last_edited_time", "url"}
            missing_fields = required_fields - set(first_page.keys())
            if missing_fields:
                print_fail(f"Missing fields in page object: {missing_fields}")
            else:
                print_pass(f"Page schema valid. First page: '{first_page['title']}'")
            print_info("All page titles", [p["title"] for p in pages])

            # ----------------------------------------------------------
            # Test 3: notion_fetch_page
            # ----------------------------------------------------------
            print_header("TEST 3: notion_fetch_page")

            target_page_id = first_page["page_id"]
            print(f"  Fetching page: '{first_page['title']}' (ID: {target_page_id})")

            fetch_result = await session.call_tool(
                "notion_fetch_page", {"page_id": target_page_id}
            )
            doc = parse_dict_result(fetch_result)

            if not doc.get("content"):
                print_fail("Page content is empty — block rendering may have failed")
            else:
                print_pass(f"Content fetched, length: {len(doc['content'])} chars")
                print_info("Content preview (first 300 chars)", doc["content"][:300])

            # Validate full document schema
            required_doc_fields = {"doc_id", "source", "title", "content", "metadata"}
            missing_doc_fields = required_doc_fields - set(doc.keys())
            if missing_doc_fields:
                print_fail(f"Missing fields in document schema: {missing_doc_fields}")
            else:
                print_pass("Document schema valid")
            print_info("metadata", doc.get("metadata", {}))

            # ----------------------------------------------------------
            # Test 4: drive_list_files
            # ----------------------------------------------------------
            print_header("TEST 4: drive_list_files")
            print("  (This may open a browser for Google OAuth on first run)")

            files_result = await session.call_tool("drive_list_files", {})
            files = parse_list_result(files_result)

            print_info("File count", len(files))
            if not files:
                print_fail("No files returned — check DRIVE_FOLDER_ID and OAuth credentials")
                return
            print_pass(f"{len(files)} files returned from Drive")

            required_file_fields = {"file_id", "name", "mime_type", "modified_time"}
            first_file = files[0]
            missing_file_fields = required_file_fields - set(first_file.keys())
            if missing_file_fields:
                print_fail(f"Missing fields in file object: {missing_file_fields}")
            else:
                print_pass(f"File schema valid. First file: '{first_file['name']}'")
            print_info("All file names", [f["name"] for f in files])

            # ----------------------------------------------------------
            # Test 5: drive_fetch_file
            # ----------------------------------------------------------
            print_header("TEST 5: drive_fetch_file")

            target_file_id = first_file["file_id"]
            print(f"  Fetching file: '{first_file['name']}' (ID: {target_file_id})")

            drive_fetch_result = await session.call_tool(
                "drive_fetch_file", {"file_id": target_file_id}
            )
            drive_doc = parse_dict_result(drive_fetch_result)

            if not drive_doc.get("content"):
                print_fail("File content is empty — MIME type routing or export may have failed")
            else:
                print_pass(f"Content fetched, length: {len(drive_doc['content'])} chars")
                print_info("Content preview (first 300 chars)", drive_doc["content"][:300])

            required_doc_fields = {"doc_id", "source", "title", "content", "metadata"}
            missing_drive_fields = required_doc_fields - set(drive_doc.keys())
            if missing_drive_fields:
                print_fail(f"Missing fields in document schema: {missing_drive_fields}")
            else:
                print_pass("Document schema valid")
            print_info("metadata", drive_doc.get("metadata", {}))

            # ----------------------------------------------------------
            # Summary
            # ----------------------------------------------------------
            print_header("ALL TESTS COMPLETE")
            print("  If all 5 tests show [OK], Phase 1 is working end-to-end.")
            print("  If any show [FAIL], see error above and check PROJECT_LOG.md for known issues.")
            print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Unhandled error: {type(e).__name__}: {e}")
        raise
