"""
query_parser.py
---------------
Parses a natural language query into the structured JSON format:

    {
      "satisfied":             bool,
      "reason":                str | None,
      "category":              str | None,
      "model":                 str | None,
      "color":                 str | None,
      "budget":                float | None,
      "clarification_question": str | None
    }

Two modes:
  1. Rule-based (default, no API needed) — regex + keyword matching.
  2. Anthropic API — if ANTHROPIC_API_KEY is set, uses Claude to parse.

The parsed fields are used to:
  - Augment the search query (category + color added to query string)
  - Post-filter results (price <= budget)
"""

import re
import os
import json


# Known categories and keywords that map to them
CATEGORY_MAP = {
    "laptop":     "Laptops",
    "notebook":   "Laptops",
    "macbook":    "Laptops",
    "phone":      "Smartphones",
    "smartphone": "Smartphones",
    "iphone":     "Smartphones",
    "android":    "Smartphones",
    "headphone":  "Audio",
    "headset":    "Audio",
    "earphone":   "Audio",
    "airpods":    "Audio",
    "speaker":    "Audio",
    "gaming":     "Gaming",
    "console":    "Gaming",
    "playstation": "Gaming",
    "xbox":       "Gaming",
    "nintendo":   "Gaming",
    "mouse":      "Gaming",
    "tv":         "TVs",
    "television": "TVs",
    "oled":       "TVs",
    "camera":     "Cameras",
    "drone":      "Cameras",
    "mirrorless": "Cameras",
    "tablet":     "Tablets",
    "ipad":       "Tablets",
    "watch":      "Wearables",
    "smartwatch": "Wearables",
    "monitor":    "Monitors",
    "display":    "Monitors",
    "charger":    "Accessories",
    "powerbank":  "Accessories",
    "cable":      "Accessories",
}

COLORS = ["black","white","silver","red","blue","green","gray","grey",
          "gold","pink","titanium","graphite","platinum","midnight","hazel"]


def parse_rule_based(query: str) -> dict:
    """
    Extract structured fields from a query using regex and keyword lookup.
    Fast, offline, no API required.
    """
    text = query.lower()

    # Category
    category = None
    for keyword, cat in CATEGORY_MAP.items():
        if keyword in text:
            category = cat
            break

    # Color
    color = None
    for c in COLORS:
        if c in text:
            color = c
            break

    # Budget — match patterns like "$1200", "under 800", "max $500", "1000 dollars"
    budget = None
    patterns = [
        r"under\s*\$?([\d,]+)",
        r"below\s*\$?([\d,]+)",
        r"max\s*\$?([\d,]+)",
        r"budget\s*of\s*\$?([\d,]+)",
        r"\$?([\d,]+)\s*(?:dollars?|usd|€|£)?",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            budget = float(m.group(1).replace(",", ""))
            break

    # Model — look for known brand names
    brands = ["apple","samsung","sony","dell","asus","lenovo","hp","google",
              "microsoft","nintendo","logitech","bose","jabra","anker","dji",
              "canon","lg","oneplug","oneplubs"]
    model = None
    for b in brands:
        if b in text:
            model = b.capitalize()
            break

    return {
        "satisfied":             True,
        "reason":                None,
        "category":              category,
        "model":                 model,
        "color":                 color,
        "budget":                budget,
        "clarification_question": None,
    }


def parse_with_api(query: str) -> dict:
    """
    Use the Anthropic API to parse the query into structured JSON.
    Falls back to rule-based if the API is unavailable.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()

        system = """You are a query parser for an e-commerce search system.
Parse the user query and return ONLY valid JSON with no other text.
Schema:
{
  "satisfied": true,
  "reason": null,
  "category": "<one of: Laptops|Smartphones|Audio|Gaming|TVs|Cameras|Tablets|Wearables|Monitors|Accessories|null>",
  "model": "<brand name or null>",
  "color": "<color or null>",
  "budget": <max price as number or null>,
  "clarification_question": "<question if query is ambiguous, else null>"
}
Return ONLY the JSON object. No markdown. No explanation."""

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": query}],
        )
        raw = message.content[0].text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)

    except Exception as e:
        print(f"[Parser] API unavailable ({e}), using rule-based fallback.")
        return parse_rule_based(query)


def parse_query(query: str) -> dict:
    """
    Main entry point. Uses API if ANTHROPIC_API_KEY is in environment,
    otherwise falls back to rule-based parser.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return parse_with_api(query)
    return parse_rule_based(query)


if __name__ == "__main__":
    tests = [
        "I need a black gaming laptop under $1300",
        "best noise cancelling headphones for travel",
        "flagship smartphone with great camera",
        "4K TV under 1500 for PS5 gaming",
        "cheap wireless mouse",
    ]
    for q in tests:
        result = parse_rule_based(q)
        print(f"\nQuery : {q}")
        print(f"Parsed: {json.dumps(result, indent=2)}")
