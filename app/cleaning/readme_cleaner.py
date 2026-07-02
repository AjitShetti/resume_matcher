"""
app/cleaning/readme_cleaner.py — Strip markdown noise from GitHub READMEs.

Preserves semantic content (prose, bullet text, heading text) while removing
visual/formatting markup that would pollute embedding vectors.

Rules applied in order:
  1. Remove fenced code blocks (``` ... ```) including their content
  2. Remove badge lines  [![...](...)](...)  — common README noise
  3. Strip markdown heading markers (#, ##, …) but keep the heading text
  4. Strip inline emphasis markers (**, *, __,  _) but keep enclosed text
  5. Convert bullet markers ("- " / "* " / "+ ") to plain text (keep text)
  6. Drop standalone decorator lines (---, ===, ***) with no alphanumeric chars
  7. Collapse multiple consecutive blank lines into one
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Compiled patterns  (compiled once at module load, not on every call)
# ---------------------------------------------------------------------------

# Fenced code blocks: opening fence can be ``` or ~~~ with optional language tag
_FENCED_CODE_BLOCK = re.compile(
    r"```[^\n]*\n.*?```|~~~[^\n]*\n.*?~~~",
    re.DOTALL,
)

# Badge lines: [![alt](img_url)](link_url)  — full-line badge combos
# Also catches bare image links: ![alt](url)
_BADGE_LINE = re.compile(
    r"^\s*(?:\[!\[.*?\]\(.*?\)\]\(.*?\)|!\[.*?\]\(.*?\))\s*$",
    re.MULTILINE,
)

# Heading markers: leading # symbols (with optional trailing #)
# Keep the text that follows — e.g. "## Installation" → "Installation"
_HEADING_MARKER = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# Bold/italic emphasis: **, *, __, _
# Order matters: match longer patterns first to avoid leaving stray single markers
_EMPHASIS = re.compile(r"\*{1,3}|_{1,3}")

# Bullet point markers at line start: "- ", "* ", "+ " (with optional leading space)
_BULLET_MARKER = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)

# Decorator-only lines: lines that contain no alphanumeric characters at all
# Catches ----, ====, ****, ~~~, etc.
_DECORATOR_LINE = re.compile(r"^[^a-zA-Z0-9\n]*$", re.MULTILINE)

# Multiple consecutive blank lines
_MULTI_BLANK = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_readme(raw_text: str) -> str:
    """
    Strip markdown noise from a raw GitHub README string.

    Returns plain text with:
      - code blocks removed entirely
      - badges removed
      - heading text preserved, markers removed
      - emphasis text preserved, markers removed
      - bullet text preserved, leading marker removed
      - decorator lines (---,  ===, …) removed
      - consecutive blank lines collapsed to one
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Remove fenced code blocks (including their content)
    text = _FENCED_CODE_BLOCK.sub("", text)

    # 2. Remove badge/image lines
    text = _BADGE_LINE.sub("", text)

    # 3. Strip heading markers, keep heading text
    text = _HEADING_MARKER.sub("", text)

    # 4. Strip emphasis markers, keep enclosed text
    text = _EMPHASIS.sub("", text)

    # 5. Strip bullet markers, keep bullet text
    text = _BULLET_MARKER.sub("", text)

    # 6. Remove decorator-only lines (lines with no alphanumeric content)
    text = "\n".join(
        line for line in text.splitlines()
        if re.search(r"[a-zA-Z0-9]", line) or line.strip() == ""
    )

    # 7. Collapse 3+ consecutive newlines → 2 (one blank line)
    text = _MULTI_BLANK.sub("\n\n", text)

    return text.strip()
