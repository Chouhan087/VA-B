"""
Tool calling for VOXIA AI.

Rather than relying on a specific model's native function-calling support
(inconsistent across Ollama models, and unusable at all in the mock-fallback
path), tools here are detected with simple, conservative pattern matching
against the user's message and executed directly in Python. The result is
then injected into the LLM call as context (see llm.py's tool_context
param) so the model phrases a natural answer around a real, computed
result — the same "ground the model in something real" pattern already
used for RAG and long-term memory.

This keeps tool calling fully testable and deterministic (same detection
logic runs whether or not Ollama is connected), at the cost of only
recognizing the phrasings coded below rather than understanding open-ended
intent the way a model with native tool-calling would.

Tools:
  - calculator: local arithmetic, no network, always available
  - reminders (add/list): backed by the Reminder table, always available
  - weather: calls Open-Meteo (free, no API key) — network-dependent;
    see README for a note on why this one couldn't be tested live here
"""

import ast
import operator
import re

import httpx

CALCULATOR_PATTERN = re.compile(
    r"(?:what(?:'s| is)|calculate|compute)\s+(?:the\s+)?(?:result\s+of\s+)?([0-9\s\.\+\-\*/\(\)%]+)\??\s*$",
    re.IGNORECASE,
)
BARE_EXPRESSION_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?\s*[\+\-\*/%]\s*[0-9\.\+\-\*/\(\)%\s]+)\??\s*$")

REMINDER_ADD_PATTERN = re.compile(
    r"(?:remind me to|add a reminder to|set a reminder to|remember to)\s+(.+)", re.IGNORECASE
)
REMINDER_LIST_PATTERN = re.compile(
    r"(?:what are my reminders|list my reminders|show (?:me )?my reminders|my reminders\??)", re.IGNORECASE
)

WEATHER_PATTERN = re.compile(
    r"(?:what'?s the weather|weather (?:like )?)(?:like )?(?:in|for|at)\s+([a-zA-Z ,.'-]{2,60})\??", re.IGNORECASE
)

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def safe_calculate(expression: str) -> float:
    """
    Evaluates a plain arithmetic expression (+ - * / % and parentheses
    only) without using eval(). Raises ValueError on anything else —
    no names, no attribute access, no function calls, nothing unsafe.
    """
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree.body)


async def fetch_weather(location: str) -> dict:
    """
    Free, keyless weather via Open-Meteo: geocode the place name, then
    pull current conditions. Raises on any failure (unknown place,
    network issue) — caller treats that as "tool didn't fire".
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        geo_resp = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results")
        if not results:
            raise ValueError(f"Couldn't find a location matching '{location}'")
        place = results[0]

        weather_resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,wind_speed_10m,relative_humidity_2m",
            },
        )
        weather_resp.raise_for_status()
        current = weather_resp.json().get("current", {})

        return {
            "place": f"{place['name']}, {place.get('country', '')}".strip(", "),
            "temperature_c": current.get("temperature_2m"),
            "wind_kmh": current.get("wind_speed_10m"),
            "humidity_pct": current.get("relative_humidity_2m"),
        }


def detect_tool(message: str) -> dict:
    """
    Returns {"tool": str, **params} if the message matches a known tool
    pattern, else {"tool": None}. Detection order matters slightly (most
    specific first) to avoid one pattern shadowing another.
    """
    reminder_add = REMINDER_ADD_PATTERN.search(message)
    if reminder_add:
        return {"tool": "reminder_add", "content": reminder_add.group(1).strip().rstrip(".!?")}

    if REMINDER_LIST_PATTERN.search(message):
        return {"tool": "reminder_list"}

    weather_match = WEATHER_PATTERN.search(message)
    if weather_match:
        return {"tool": "weather", "location": weather_match.group(1).strip().rstrip(".!?")}

    calc_match = CALCULATOR_PATTERN.search(message) or BARE_EXPRESSION_PATTERN.match(message)
    if calc_match:
        return {"tool": "calculator", "expression": calc_match.group(1).strip()}

    return {"tool": None}
