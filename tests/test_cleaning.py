"""
tests/test_cleaning.py — Tests for app.cleaning.readme_cleaner.

Run standalone:  python -m tests.test_cleaning
"""

import os
os.environ.setdefault("GITHUB_USERNAME", "AjitShetti")

from app.cleaning.readme_cleaner import clean_readme  # noqa: E402

# ---------------------------------------------------------------------------
# Shared sample README used across tests
# ---------------------------------------------------------------------------

SAMPLE_README = """\
[![Build Status](https://travis-ci.org/user/repo.svg?branch=main)](https://travis-ci.org/user/repo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)

# My Awesome Project

A **FastAPI** backend that uses _machine learning_ to match resumes with job descriptions.

---

## Features

- Semantic similarity scoring with **sentence-transformers**
- REST API built on *FastAPI*
- Automated README cleaning

===

## Installation

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Usage

```python
import requests
response = requests.post("/match", json={"jd_text": "..."})
print(response.json())
```

Some closing prose with __important__ concepts and ***triple emphasis***.

Multiple


blank


lines above should collapse.
"""

# ---------------------------------------------------------------------------
# Individual rule tests
# ---------------------------------------------------------------------------

def test_badges_removed() -> None:
    """Badge lines ([![...](...)](...) and ![...](url)) must be gone."""
    result = clean_readme(SAMPLE_README)
    assert "travis-ci.org" not in result, "Travis CI badge URL should be removed"
    assert "img.shields.io" not in result, "Shields.io badge URL should be removed"
    assert "[![" not in result, "Badge syntax [![ should be gone"
    print("PASS: badges removed")


def test_code_blocks_removed() -> None:
    """Fenced code blocks including their content must be gone."""
    result = clean_readme(SAMPLE_README)
    assert "pip install" not in result, "Code block content (pip install) should be gone"
    assert "uvicorn" not in result, "Code block content (uvicorn) should be gone"
    assert "import requests" not in result, "Code block content (import requests) should be gone"
    assert "```" not in result, "Fenced code fence markers should be gone"
    print("PASS: code blocks removed")


def test_heading_markers_removed_text_kept() -> None:
    """# markers stripped, heading text preserved."""
    result = clean_readme(SAMPLE_README)
    assert "# My Awesome Project" not in result, "Raw heading with # should be gone"
    assert "My Awesome Project" in result, "Heading text must be preserved"
    assert "Features" in result, "## Features text must be preserved"
    assert "Installation" in result, "## Installation text must be preserved"
    print("PASS: heading markers removed, text kept")


def test_emphasis_markers_removed_text_kept() -> None:
    """**, *, __, _ markers stripped, enclosed text preserved."""
    result = clean_readme(SAMPLE_README)
    assert "**FastAPI**" not in result, "**bold** syntax should be gone"
    assert "_machine learning_" not in result, "_italic_ syntax should be gone"
    assert "FastAPI" in result, "FastAPI text must be preserved"
    assert "machine learning" in result, "machine learning text must be preserved"
    assert "important" in result, "__important__ text must be preserved"
    assert "triple emphasis" in result, "***triple*** text must be preserved"
    print("PASS: emphasis markers removed, text kept")


def test_bullet_markers_removed_text_kept() -> None:
    """Leading '- ' stripped, bullet text preserved."""
    result = clean_readme(SAMPLE_README)
    # The bullet text should be present without its leading dash
    assert "Semantic similarity scoring with" in result, "Bullet text must be preserved"
    assert "REST API built on" in result, "Bullet text must be preserved"
    assert "Automated README cleaning" in result, "Bullet text must be preserved"
    # Check that no line starts with "- " (bullet marker)
    for line in result.splitlines():
        assert not line.startswith("- "), f"Bullet marker still present in line: {line!r}"
    print("PASS: bullet markers removed, text kept")


def test_decorator_lines_removed() -> None:
    """Pure punctuation/symbol lines (---, ===) must be gone."""
    result = clean_readme(SAMPLE_README)
    for line in result.splitlines():
        stripped = line.strip()
        if stripped:  # ignore blank lines
            import re
            assert re.search(r"[a-zA-Z0-9]", stripped), (
                f"Decorator-only line survived: {line!r}"
            )
    print("PASS: decorator lines removed")


def test_blank_lines_collapsed() -> None:
    """Multiple consecutive blank lines collapsed to a single blank line."""
    result = clean_readme(SAMPLE_README)
    assert "\n\n\n" not in result, "Triple newline should have been collapsed"
    print("PASS: multiple blank lines collapsed")


def test_empty_input() -> None:
    """Empty / None-like input returns empty string without crashing."""
    assert clean_readme("") == ""
    print("PASS: empty input handled")


def test_prose_preserved() -> None:
    """Non-markdown prose sentences survive unchanged (modulo emphasis stripping)."""
    result = clean_readme(SAMPLE_README)
    assert "A FastAPI backend that uses" in result, "Prose sentence must survive"
    assert "match resumes with job descriptions" in result, "Prose must survive"
    print("PASS: prose preserved")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Stage 3 — readme_cleaner tests")
    print("=" * 60 + "\n")

    test_badges_removed()
    test_code_blocks_removed()
    test_heading_markers_removed_text_kept()
    test_emphasis_markers_removed_text_kept()
    test_bullet_markers_removed_text_kept()
    test_decorator_lines_removed()
    test_blank_lines_collapsed()
    test_empty_input()
    test_prose_preserved()

    print("\n--- Cleaned output preview ---\n")
    print(clean_readme(SAMPLE_README))
    print("\nAll tests passed.")
