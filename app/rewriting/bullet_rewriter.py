"""
app/rewriting/bullet_rewriter.py — LLM-powered resume bullet rewriter.

Calls the Groq API (free tier) to reframe existing resume bullets using JD
vocabulary, with a strict anti-hallucination prompt: every fact must come
from the README.

Safety guarantees:
- The model is instructed never to invent technologies, metrics, or claims.
- If the API call fails for any reason, original bullets are returned unchanged.
- format_review() provides a mandatory side-by-side diff before any bullet is accepted.

Get a free API key at: https://console.groq.com
"""

from __future__ import annotations

import logging
import os

from groq import Groq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a professional resume writer. Your job is to rewrite resume project \
bullet points so they resonate with a specific job description, while remaining \
100% factually accurate.

STRICT RULES — violating any of these is unacceptable:
1. Use ONLY the provided README text as the source of truth for facts, \
technologies, tools, metrics, and achievements.
2. You MAY reframe, reprioritize, and reword using vocabulary from the job \
description to improve relevance.
3. NEVER introduce any technology, tool, metric, scale claim, or achievement \
that is not explicitly present in the README. If the JD emphasises something \
the README does not support, do not force the connection — simply omit it.
4. Return ONLY the rewritten bullet points — one per line, no preamble, \
no explanations, no numbering, no dashes.
5. Produce the same number of bullets as provided in the existing list \
where possible.
"""

_USER_TEMPLATE = """\
README (source of truth — do not invent facts beyond this):
\"\"\"
{readme_text}
\"\"\"

Job Description (use its vocabulary and framing, but only where the README supports it):
\"\"\"
{jd_text}
\"\"\"

Existing resume bullets for this project (improve these, do not replace from scratch):
{bullets_formatted}

Rewrite the bullets now. Return only the bullet text, one per line.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_bullets(
    readme_text: str,
    jd_text: str,
    existing_bullets: list[str],
) -> list[str]:
    """
    Rewrite resume bullets using the Anthropic API.

    Falls back to returning the original bullets unchanged if the API call
    fails for any reason (network error, rate limit, invalid key, etc.).

    Args:
        readme_text:      Raw (or cleaned) README text — the only allowed fact source.
        jd_text:          Job description text used for vocabulary/framing.
        existing_bullets: Current resume bullets for this project.

    Returns:
        List of rewritten bullet strings, same length as input where possible.
    """
    bullets_formatted = "\n".join(f"- {b}" for b in existing_bullets)

    user_prompt = _USER_TEMPLATE.format(
        readme_text=readme_text.strip(),
        jd_text=jd_text.strip(),
        bullets_formatted=bullets_formatted,
    )

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        rewritten = [line.strip() for line in raw.splitlines() if line.strip()]
        return rewritten

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Groq API call failed — returning original bullets unchanged. "
            "Error: %s",
            exc,
        )
        return list(existing_bullets)


def format_review(
    original_bullets: list[str],
    rewritten_bullets: list[str],
) -> str:
    """
    Format a side-by-side text comparison of original vs rewritten bullets.

    This is a required manual checkpoint — review this output before accepting
    any rewritten bullets into your resume.

    Args:
        original_bullets:  The bullets before rewriting.
        rewritten_bullets: The bullets after rewriting.

    Returns:
        A formatted multi-line string for human review.
    """
    lines: list[str] = [
        "=" * 70,
        "  BULLET REVIEW — Original vs Rewritten",
        "  Review carefully before accepting. Reject any invented facts.",
        "=" * 70,
    ]

    max_len = max(len(original_bullets), len(rewritten_bullets))

    for i in range(max_len):
        orig = original_bullets[i] if i < len(original_bullets) else "(no original)"
        new  = rewritten_bullets[i] if i < len(rewritten_bullets) else "(no rewrite)"

        lines.append(f"\n[{i + 1}] ORIGINAL : {orig}")
        lines.append(f"    REWRITTEN: {new}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
