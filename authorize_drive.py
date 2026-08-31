"""
authorize_drive.py -- One-time Google Drive OAuth authorization helper.

Run this script ONCE in a terminal where a browser can open:
    .venv\\Scripts\\python authorize_drive.py

It will:
  1. Open your browser to Google's OAuth consent page
  2. Ask you to sign in and click Allow
  3. Cache the result in token.json (next to credentials.json)

After this script succeeds, token.json exists and the test script /
MCP server will NEVER need to open a browser again for Drive access.

This script is intentionally separate from the MCP server so that
the one-time browser interaction can happen in a normal terminal
session where a browser is available.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv(override=False)

# Add project root to path so relative package imports work if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

creds_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "credentials.json")
creds_path = os.path.abspath(creds_path)
token_path = os.path.join(os.path.dirname(creds_path), "token.json")

print(f"credentials.json : {creds_path}")
print(f"token.json target: {token_path}")
print()

if os.path.exists(token_path):
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if creds and creds.valid:
        print("[OK] token.json already exists and is valid. No browser needed.")
        print("     You can now run: .venv\\Scripts\\python test_mcp_connection.py")
        sys.exit(0)

print("Opening browser for Google OAuth consent...")
print("(If browser does not open automatically, copy the URL printed below into your browser)")
print()

flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
creds = flow.run_local_server(port=0, open_browser=True)

with open(token_path, "w") as f:
    f.write(creds.to_json())

print()
print(f"[OK] token.json saved to: {token_path}")
print("     Drive authorization complete. Run the test now:")
print("     .venv\\Scripts\\python test_mcp_connection.py")
