"""
Guardrails: prompt injection detection + off-topic scope classification.

Design: Two-layer defence.
  Layer 1 (fast, deterministic): Regex/keyword detection for known injection patterns.
  Layer 2 (LLM-based): Pass the last user message through a zero-shot classifier
                        with a restricted system prompt.  Only invoked if layer 1 passes.

Scope categories:
  - "shl_assessment"   → proceed normally
  - "off_topic"        → polite refusal
  - "injection"        → firm refusal
  - "comparison"       → comparison branch
  - "clarification"    → agent should continue gathering info
"""

import re
import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer 1: deterministic pattern matching
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|all|above)\s+(instructions?|rules?|prompts?)",
    r"disregard\s+.{0,30}\s*(instructions?|rules?)",
    r"you\s+are\s+now\s+(a|an)\s+\w",          # "you are now a DAN"
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"print\s+(your\s+)?(system\s+)?prompt",
    r"bypass\s+(safety|filter|guardrail)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"pretend\s+(you\s+are|to\s+be)",
    r"roleplay\s+as",
    r"act\s+as\s+(if\s+you\s+are\s+)?(?!an?\s+shl)",
    r"forget\s+(everything|all|prior)",
    r"new\s+instructions?\s*:",
    r"override\s+(your\s+)?(instructions?|programming)",
]

OFF_TOPIC_PATTERNS = [
    r"\blegal(ly)?\b.*\b(fire|dismiss|terminat|sue|liab)",
    r"\b(fire|dismiss|terminat)\b.*\blegal",
    r"\blegal\s+(advice|counsel|question)\b",
    r"\blawsuit\b",
    r"\blitigat\b",
    r"\bmedical\s+advice\b",
    r"\bstock\s+pick\b",
    r"\bcrypto\b",
    r"\brecipe\b",
    r"\bweather\b",
    r"\bsports\b(?!\s+assessment)",
    r"\bpolitics\b",
    r"\bcelebrit",
    r"\bwrite\s+(me\s+)?(a\s+)?(poem|story|essay|code(?!\s+test))",
    r"\btranslate\s+(this|the)\b",
    r"\bsummarise\s+(this\s+)?(article|document|text)\b",
    r"\bwhat\s+is\s+the\s+weather\b",
    r"\bwho\s+(is|was)\s+\w+\s+\w+\b",           # "who is Elon Musk" style
    r"\bgeneral\s+hr\s+advice\b",
    r"\bhow\s+to\s+fire\s+someone\b",
    r"\bnon.?shl\b",
]

COMPARISON_PATTERNS = [
    r"\bdifference\s+between\b",
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bbetter\s+(than|for)\b",
    r"\bwhich\s+(is|one)\s+(better|best|more)\b",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_OFF_TOPIC_RE = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]
_COMPARISON_RE = [re.compile(p, re.IGNORECASE) for p in COMPARISON_PATTERNS]


def classify_message(text: str) -> str:
    """
    Fast deterministic classifier.
    Returns: "injection" | "off_topic" | "comparison" | "shl_assessment"
    """
    for pattern in _INJECTION_RE:
        if pattern.search(text):
            log.warning("Injection pattern detected: %.80s", text)
            return "injection"

    for pattern in _OFF_TOPIC_RE:
        if pattern.search(text):
            log.info("Off-topic pattern detected: %.80s", text)
            return "off_topic"

    for pattern in _COMPARISON_RE:
        if pattern.search(text):
            return "comparison"

    return "shl_assessment"


REFUSAL_INJECTION = (
    "I'm here to help with SHL assessment selection only. "
    "I can't follow instructions that ask me to override my purpose or reveal system configuration."
)

REFUSAL_OFF_TOPIC = (
    "I'm focused on SHL assessment recommendations. "
    "For that question, you'd be better served by a different resource. "
    "How can I help you find the right SHL assessment?"
)