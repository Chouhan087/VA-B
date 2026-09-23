"""
Long-term memory extraction for VOXIA AI.

After each user message, decide whether it contains a durable fact worth
remembering across separate conversations — identity, ongoing goals,
preferences, constraints — as opposed to a one-off remark or question.
Extracted facts get embedded and stored (see rag.py for the embedding
machinery); at the start of future chat turns, main.py retrieves whichever
stored facts are relevant to the current message the same way it retrieves
document chunks.

Same two-path pattern as llm.py and rag.py: a real LLM call when Ollama is
available, a deterministic fallback (regex over common self-descriptive
phrasings) when it isn't, so the whole feature is testable with zero setup.
"""

import json
import re

from llm import complete_text

EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable, worth-remembering facts about the user from a single "
    "chat message, for a personal AI assistant's long-term memory. Only extract "
    "facts that would still be true or relevant weeks from now: identity "
    "(name, role, goals), ongoing situations (exams, projects, deadlines), "
    "preferences, or constraints. Do NOT extract one-off questions, small talk, "
    "or facts about anything other than the user themselves.\n\n"
    "Respond with ONLY a JSON array of short strings, each a standalone fact "
    "phrased in the third person (e.g. [\"Is preparing for a DBMS exam\"]). "
    "If there is nothing worth remembering, respond with exactly: []\n"
    "No other text, no markdown fences."
)


async def _extract_via_openrouter(user_message: str) -> list:
    content = await complete_text(EXTRACTION_SYSTEM_PROMPT, user_message, max_tokens=220)
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    facts = json.loads(content)
    if not isinstance(facts, list):
        return []
    return [str(f).strip()[:400] for f in facts if isinstance(f, (str, int, float)) and str(f).strip()][:8]


# Deterministic fallback: a handful of common self-descriptive phrasings.
# Deliberately conservative — false negatives (missing a fact) are fine,
# false positives (inventing one) are not.
_FALLBACK_PATTERNS = [
    (re.compile(r"\bmy name is ([a-zA-Z ,.'-]{2,40})", re.I), "Name is {}"),
    (re.compile(r"\bi(?:'m| am) preparing for ([a-zA-Z0-9 ,.'-]{2,60})", re.I), "Is preparing for {}"),
    (re.compile(r"\bi(?:'m| am) studying ([a-zA-Z0-9 ,.'-]{2,60})", re.I), "Is studying {}"),
    (re.compile(r"\bi(?:'m| am) learning ([a-zA-Z0-9 ,.'-]{2,60})", re.I), "Is learning {}"),
    (re.compile(r"\bi work as (?:an? )?([a-zA-Z0-9 ,.'-]{2,60})", re.I), "Works as {}"),
    (re.compile(r"\bi live in ([a-zA-Z0-9 ,.'-]{2,60})", re.I), "Lives in {}"),
    (re.compile(r"\bi(?:'m| am) a[n]? ([a-zA-Z0-9 ,.'-]{2,60})", re.I), "Is a {}"),
    (re.compile(r"\bmy exam is (?:on |in )?([a-zA-Z0-9 ,.'-]{2,60})", re.I), "Has an exam {}"),
    (re.compile(r"\bi prefer ([a-zA-Z0-9 ,.'-]{2,60})", re.I), "Prefers {}"),
    (re.compile(r"\bi(?:'m| am) working on ([a-zA-Z0-9 ,.'-]{2,60})", re.I), "Is working on {}"),
]


def _extract_via_fallback(user_message: str) -> list:
    facts = []
    for pattern, template in _FALLBACK_PATTERNS:
        match = pattern.search(user_message)
        if match:
            value = match.group(1).strip().rstrip(".,!?")
            if value:
                facts.append(template.format(value))
    return facts


async def extract_memories(user_message: str) -> dict:
    """
    Returns {"facts": list[str], "source": "openrouter" | "fallback"}
    """
    try:
        facts = await _extract_via_openrouter(user_message)
        return {"facts": facts, "source": "openrouter"}
    except Exception:
        return {"facts": _extract_via_fallback(user_message), "source": "fallback"}
