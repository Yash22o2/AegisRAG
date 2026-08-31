"""
drive_connector.py — Google Drive API wrapper for AegisRAG MCP server.

Why this file exists:
    Isolates all Drive-specific logic (OAuth flow, file listing, content
    extraction per MIME type) from server.py. The MCP tools call exactly two
    functions from here:
        list_files()       → list of dicts (metadata only, no content)
        fetch_file()       → NormalizedDocument with extracted text content

Design decisions:
    - OAuth 2.0 Desktop flow (InstalledAppFlow): appropriate for a personal
      portfolio project. First run opens a browser for consent; subsequent runs
      use the cached token.json. A service account would require a Workspace
      org — not available for a personal Gmail setup.
    - MIME type routing for content extraction:
          Google Docs    → export as text/plain via the export endpoint
          Google Sheets  → export as text/csv (preserves table structure better
                           than text/plain, which collapses all formatting)
          Google Slides  → export as text/plain
          PDF            → download binary, extract text with pypdf
          Other          → attempt text/plain export; fallback note if unsupported
    - Folder scoping: all list/fetch operations are scoped to DRIVE_FOLDER_ID
      (set in .env) — the connector never touches files outside that folder.
      This is important for both safety and relevance: we only want Royal
      Industries' 5 Drive documents in the corpus.
    - token.json is stored in the project root (next to credentials.json) and
      is gitignored. This avoids hardcoding a path and keeps creds together.

Environment variables consumed (loaded by caller via python-dotenv):
    GOOGLE_CREDENTIALS_PATH   — Path to credentials.json (relative or absolute)
    DRIVE_FOLDER_ID           — The Drive folder to scope file listing to
"""

from __future__ import annotations

import csv
import io
import os
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from .normalize import make_document, NormalizedDocument


# OAuth scopes — read-only Drive access is sufficient for ingestion
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# MIME types returned by the Drive API for Google Workspace native formats
MIME_GOOGLE_DOC    = "application/vnd.google-apps.document"
MIME_GOOGLE_SHEET  = "application/vnd.google-apps.spreadsheet"
MIME_GOOGLE_SLIDES = "application/vnd.google-apps.presentation"
MIME_PDF           = "application/pdf"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_credentials() -> Credentials:
    """
    Obtain (or refresh/generate) Google OAuth credentials.

    Logic:
        1. If token.json exists and is valid → use it directly.
        2. If token.json exists but is expired and has a refresh token → refresh.
        3. Otherwise → run the InstalledAppFlow (opens browser for consent).

    token.json is written/updated after every successful auth so subsequent
    runs don't need browser interaction.

    The token.json is stored in the same directory as credentials.json (project
    root) — keeps all credential files together and makes gitignoring simple.
    """
    creds_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    # Resolve relative paths against the CWD (project root when running server.py)
    creds_path = os.path.abspath(creds_path)
    token_path = os.path.join(os.path.dirname(creds_path), "token.json")

    creds: Optional[Credentials] = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            # port=0 → OS picks a free port, avoids conflicts with other local services
            creds = flow.run_local_server(port=0)

        # Persist refreshed/new credentials
        with open(token_path, "w") as token_file:
            token_file.write(creds.to_json())

    return creds


def _get_service():
    """Build and return an authenticated Google Drive API service object."""
    creds = _get_credentials()
    return build("drive", "v3", credentials=creds)


def _export_google_doc_as_text(service, file_id: str) -> str:
    """Export a Google Doc as plain text via the Drive export endpoint."""
    response = service.files().export(fileId=file_id, mimeType="text/plain").execute()
    # export() returns bytes
    return response.decode("utf-8", errors="replace")


def _export_google_sheet_as_csv(service, file_id: str) -> str:
    """
    Export a Google Sheet as CSV text.

    Design note: Google Sheets with multiple tabs export only the FIRST tab
    when using text/csv. The Royal Industries budget doc has:
        Sheet1 — structured table (Category / Allocated INR / % of Total / Notes)
        Sheet2 — prose narrative

    To capture both tabs, we export as text/plain (which concatenates all sheets
    with sheet-name separators) as a second pass. The primary format is CSV for
    the table; text/plain is appended for completeness.

    This is one of the "anticipated issues" flagged in PROJECT_LOG.md — addressed
    here with a two-pass approach rather than losing the Sheet2 narrative.
    """
    # Pass 1: CSV of Sheet1 (the table)
    csv_bytes = service.files().export(fileId=file_id, mimeType="text/csv").execute()
    csv_text = csv_bytes.decode("utf-8", errors="replace")

    # Pass 2: text/plain of all sheets (concatenated by Drive)
    try:
        plain_bytes = service.files().export(fileId=file_id, mimeType="text/plain").execute()
        plain_text = plain_bytes.decode("utf-8", errors="replace")
        # Return both: CSV table first, then the prose
        return f"[Sheet1 — CSV format]\n{csv_text}\n\n[All sheets — plain text]\n{plain_text}"
    except Exception:
        # If text/plain export fails, just return the CSV
        return csv_text


def _extract_pdf_text(service, file_id: str) -> str:
    """
    Download a PDF from Drive and extract its text content with pypdf.

    Why pypdf (not pdfminer or pdfplumber):
        pypdf is a pure-Python library with no system dependencies, making it
        the most portable choice for a portfolio project that someone else might
        clone and run. Accuracy is sufficient for clean PDFs; if Royal Industries
        ever adds scanned PDFs, OCR (e.g., pytesseract) would be needed — a
        later improvement, not a Phase 1 concern.
    """
    import pypdf  # Lazy import so the package isn't required if no PDFs exist

    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)
    reader = pypdf.PdfReader(buffer)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def _extract_content(service, file_id: str, mime_type: str, file_name: str) -> str:
    """
    Route content extraction to the right method based on MIME type.

    Returns the extracted text. If the MIME type is unrecognized, returns a
    placeholder string so the document still appears in the corpus (with a
    clear signal that content is missing) rather than crashing the pipeline.
    """
    if mime_type == MIME_GOOGLE_DOC:
        return _export_google_doc_as_text(service, file_id)
    elif mime_type == MIME_GOOGLE_SHEET:
        return _export_google_sheet_as_csv(service, file_id)
    elif mime_type == MIME_GOOGLE_SLIDES:
        return _export_google_doc_as_text(service, file_id)
    elif mime_type == MIME_PDF:
        return _extract_pdf_text(service, file_id)
    else:
        # Attempt generic text export — works for .txt, .md, etc.
        try:
            response = service.files().export(fileId=file_id, mimeType="text/plain").execute()
            return response.decode("utf-8", errors="replace")
        except Exception:
            try:
                # Last resort: download raw bytes and decode
                request = service.files().get_media(fileId=file_id)
                buffer = io.BytesIO()
                downloader = MediaIoBaseDownload(buffer, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return buffer.getvalue().decode("utf-8", errors="replace")
            except Exception as e:
                return f"[Content extraction failed for {file_name}: {e}]"


def _infer_doc_type(name: str, mime_type: str) -> str:
    """
    Simple keyword + MIME-based doc_type inference.

    Returns one of: policy, meeting_notes, spec, checklist, budget, unknown.
    Mirrors the logic in notion_connector.py for consistency.
    """
    name_lower = name.lower()
    if "meeting" in name_lower or "notes" in name_lower:
        return "meeting_notes"
    if "spec" in name_lower or "specification" in name_lower:
        return "spec"
    if "checklist" in name_lower or "onboarding" in name_lower:
        return "checklist"
    if "budget" in name_lower or "financial" in name_lower:
        return "budget"
    if "vendor" in name_lower or "contract" in name_lower:
        return "policy"   # Contract summaries are close to policy docs for RAG purposes
    if mime_type == MIME_GOOGLE_SHEET:
        return "budget"   # All sheets in this corpus are budget/tabular
    return "unknown"


# ---------------------------------------------------------------------------
# Public API (called from server.py)
# ---------------------------------------------------------------------------

def list_files() -> list[dict]:
    """
    Return metadata for all files in the scoped Drive folder.

    Scoped to DRIVE_FOLDER_ID — only returns direct children of that folder
    (not recursive subdirectories, which don't exist in our corpus).

    Returns a list of dicts, each containing:
        file_id, name, mime_type, modified_time, web_view_link

    Pagination: Drive API also paginates at 100 files (default pageSize) —
    handled here with a nextPageToken loop.
    """
    service = _get_service()
    folder_id = os.environ.get("DRIVE_FOLDER_ID")
    if not folder_id:
        raise EnvironmentError(
            "DRIVE_FOLDER_ID is not set. Check your .env file."
        )

    file_list: list[dict] = []
    page_token = None

    while True:
        kwargs = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink, owners)",
            "pageSize": 100,
        }
        if page_token:
            kwargs["pageToken"] = page_token

        response = service.files().list(**kwargs).execute()
        file_list.extend(response.get("files", []))

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    # Shape into lightweight metadata dicts
    result = []
    for f in file_list:
        owners = f.get("owners", [])
        author = owners[0].get("displayName") if owners else None
        result.append({
            "file_id": f["id"],
            "name": f["name"],
            "mime_type": f.get("mimeType", ""),
            "modified_time": f.get("modifiedTime"),
            "web_view_link": f.get("webViewLink"),
            "author": author,
        })

    return result


def fetch_file(file_id: str) -> NormalizedDocument:
    """
    Fetch the full content of a Drive file and return a NormalizedDocument.

    Steps:
        1. Retrieve file metadata (name, mime_type, modified_time, owner, web_view_link)
        2. Extract text content via `_extract_content` (routes by MIME type)
        3. Wrap in NormalizedDocument via `make_document`
    """
    service = _get_service()

    # Step 1: file metadata
    file_meta = service.files().get(
        fileId=file_id,
        fields="id, name, mimeType, modifiedTime, webViewLink, owners"
    ).execute()

    name = file_meta.get("name", "(Untitled)")
    mime_type = file_meta.get("mimeType", "")
    modified_time = file_meta.get("modifiedTime")
    web_view_link = file_meta.get("webViewLink")
    owners = file_meta.get("owners", [])
    author = owners[0].get("displayName") if owners else None
    doc_type = _infer_doc_type(name, mime_type)

    # Step 2: content extraction
    content = _extract_content(service, file_id, mime_type, name)

    # Step 3: normalize
    return make_document(
        doc_id=file_id,
        source="drive",
        title=name,
        content=content,
        author=author,
        modified_date=modified_time,
        doc_type=doc_type,
        path_or_url=web_view_link,
    )
