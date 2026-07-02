"""
app/config.py — Configuration and constants for resume-matcher.

No business logic here. Only configuration loading and constants.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (two levels up from this file:
#   app/config.py  →  app/  →  project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Required environment variables — fail fast at import time if missing
# ---------------------------------------------------------------------------
GITHUB_USERNAME: str = os.environ.get("GITHUB_USERNAME", "")
if not GITHUB_USERNAME:
    raise EnvironmentError(
        "GITHUB_USERNAME is not set. "
        "Add it to your .env file (see .env.example)."
    )

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Scoring weights
# All three values are kept together so it's obvious they form one concept.
# ---------------------------------------------------------------------------
SCORING_WEIGHTS: dict[str, float] = {
    "description": 0.35,
    "readme": 0.65,
    "name_keyword_bonus": 0.05,
}

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
# Path to the shortlist JSON — resolved relative to this file, not the CWD,
# so it works regardless of where the process is started from.
SHORTLIST_PATH: Path = Path(__file__).resolve().parent / "github" / "shortlist.json"
