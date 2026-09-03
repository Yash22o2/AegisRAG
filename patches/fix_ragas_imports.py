"""
patches/fix_ragas_imports.py
----------------------------
Applies a compatibility patch to the installed ragas package so it works
with langchain_community >= 0.3, which removed the `chat_models.vertexai`
and reorganised several module paths.

WHY THIS EXISTS
ragas 0.4.x hard-imports ChatVertexAI from langchain_community.chat_models.vertexai
at module load time in ragas/llms/base.py. langchain_community 0.3+ removed that
module path (moved to langchain-google-vertexai). This causes an immediate
ModuleNotFoundError when you do `import ragas`, even though we never use Vertex AI.

This is a bug in ragas (the imports should be lazy/optional). Until it is fixed
upstream, this script patches the installed file in-place.

REPRODUCIBILITY
Run this once after any `pip install -r requirements.txt` that installs/upgrades ragas:

    .venv\\Scripts\\python patches/fix_ragas_imports.py

The script is idempotent — safe to run multiple times. It checks whether the patch
is already applied before modifying anything.

DETECTING IF NEEDED
If `import ragas` raises ModuleNotFoundError about langchain_community.chat_models.vertexai,
run this script. If `import ragas` works fine, this script is a no-op.

LONG-TERM FIX
When ragas ships a version that handles these imports gracefully (lazy import or
optional dependency), upgrade and remove this script. Track the ragas GitHub issue:
https://github.com/explodinggradients/ragas/issues
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path


# ── Sentinel strings that identify the broken lines ──────────────────────────

BROKEN_VERTEXAI_CHAT = "from langchain_community.chat_models.vertexai import ChatVertexAI"
BROKEN_VERTEXAI_LLM  = "from langchain_community.llms import VertexAI"
BROKEN_OPENAI        = "from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI"

# A string that only appears AFTER the patch — used to detect already-patched files
PATCH_SENTINEL = "# ragas-compat-patch: wrapped by patches/fix_ragas_imports.py"

# ── Replacement blocks ────────────────────────────────────────────────────────

REPLACEMENT_VERTEXAI_CHAT = """\
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ImportError:
    ChatVertexAI = None  # type: ignore[assignment,misc]  # ragas-compat-patch: wrapped by patches/fix_ragas_imports.py"""

REPLACEMENT_VERTEXAI_LLM = """\
try:
    from langchain_community.llms import VertexAI
except ImportError:
    VertexAI = None  # type: ignore[assignment,misc]  # ragas-compat-patch: wrapped by patches/fix_ragas_imports.py"""

REPLACEMENT_OPENAI = """\
try:
    from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
    from langchain_openai.llms import AzureOpenAI, OpenAI
    from langchain_openai.llms.base import BaseOpenAI
except ImportError:
    AzureChatOpenAI = ChatOpenAI = AzureOpenAI = OpenAI = BaseOpenAI = None  # type: ignore[assignment,misc]  # ragas-compat-patch: wrapped by patches/fix_ragas_imports.py"""

BROKEN_OPENAI_FULL = """\
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.llms import AzureOpenAI, OpenAI
from langchain_openai.llms.base import BaseOpenAI"""


def find_ragas_base() -> Path | None:
    """Locate ragas/llms/base.py in the current Python environment."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import pathlib; "
             "exec(open(pathlib.Path(__import__('ragas').__file__).parent / 'llms' / 'base.py').read(), {}); "
             "import ragas; "
             "print(pathlib.Path(ragas.__file__).parent / 'llms' / 'base.py')"],
            capture_output=True, text=True, timeout=10
        )
        # simpler approach:
        result2 = subprocess.run(
            [sys.executable, "-c",
             "import importlib.util, pathlib; "
             "spec = importlib.util.find_spec('ragas'); "
             "print(pathlib.Path(spec.origin).parent / 'llms' / 'base.py') if spec else print('')"],
            capture_output=True, text=True, timeout=10
        )
        p = result2.stdout.strip()
        if p:
            path = Path(p)
            if path.exists():
                return path
    except Exception:
        pass
    return None


def apply_patch(base_py: Path) -> None:
    """Apply the compatibility patch to ragas/llms/base.py."""
    content = base_py.read_text(encoding="utf-8")

    if PATCH_SENTINEL in content:
        print(f"[SKIP] Patch already applied to {base_py}")
        return

    modified = content

    # Patch 1: ChatVertexAI hard import
    if BROKEN_VERTEXAI_CHAT in modified:
        modified = modified.replace(BROKEN_VERTEXAI_CHAT, REPLACEMENT_VERTEXAI_CHAT)
        print(f"  [PATCH] Wrapped ChatVertexAI import in try/except")
    else:
        print(f"  [SKIP]  ChatVertexAI import not found — may already be patched or ragas changed")

    # Patch 2: VertexAI (LLM) hard import
    if BROKEN_VERTEXAI_LLM in modified:
        modified = modified.replace(BROKEN_VERTEXAI_LLM, REPLACEMENT_VERTEXAI_LLM)
        print(f"  [PATCH] Wrapped VertexAI LLM import in try/except")
    else:
        print(f"  [SKIP]  VertexAI LLM import not found")

    # Patch 3: OpenAI imports (may fail if langchain-openai not installed)
    if BROKEN_OPENAI_FULL in modified:
        modified = modified.replace(BROKEN_OPENAI_FULL, REPLACEMENT_OPENAI)
        print(f"  [PATCH] Wrapped OpenAI imports in try/except")
    else:
        print(f"  [SKIP]  OpenAI imports not found in expected form")

    if modified == content:
        print(f"[INFO] No changes needed — file looks already clean.")
        return

    base_py.write_text(modified, encoding="utf-8")
    print(f"[OK] Patch written to {base_py}")


def verify_patch() -> bool:
    """Verify ragas imports cleanly after patching."""
    result = subprocess.run(
        [sys.executable, "-c", "import ragas; print('ragas', ragas.__version__)"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        print(f"[OK] {result.stdout.strip()} imports cleanly.")
        return True
    else:
        print(f"[FAIL] ragas still fails to import after patch:")
        print(result.stderr)
        return False


def main() -> None:
    print("AegisRAG — ragas compatibility patcher")
    print("=" * 45)

    base_py = find_ragas_base()
    if base_py is None:
        print("[ERROR] Could not locate ragas installation.")
        print("  Make sure ragas is installed: pip install ragas")
        sys.exit(1)

    print(f"Found: {base_py}")

    apply_patch(base_py)

    print("\nVerifying...")
    ok = verify_patch()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
