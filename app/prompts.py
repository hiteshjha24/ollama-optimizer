"""Predefined benchmark prompts.

Each prompt carries an ``expectations`` block describing what can be checked
deterministically. Where a prompt has a verifiable answer the evaluator can
score correctness; where it does not (creative writing, open summarisation)
correctness is reported as *not scorable* rather than guessed at.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# A deterministic filler corpus used to build the long-context prompt.
_FILLER_PARAGRAPHS = [
    "The maintenance crew logs equipment readings at the start of every shift, "
    "recording vibration, temperature and lubricant pressure for each unit on the line.",
    "Warehouse inventory is reconciled weekly against the receiving ledger, and any "
    "discrepancy larger than two units is escalated to the floor supervisor.",
    "Shift handovers are documented in the operations notebook so that the incoming "
    "team inherits the open issues, pending parts orders and safety observations.",
    "Calibration certificates for the measurement rigs are renewed annually and stored "
    "with the quality records in the east office filing cabinet.",
    "Energy consumption for the compressor hall is metered separately from the main "
    "building supply, which makes seasonal comparisons more reliable.",
]


def _long_context_document(repeats: int = 14) -> str:
    """Build a long document with one verifiable fact hidden inside it."""
    blocks: List[str] = []
    for index in range(repeats):
        blocks.append(f"Section {index + 1}.")
        for paragraph in _FILLER_PARAGRAPHS:
            blocks.append(paragraph)
        if index == repeats // 2:
            blocks.append(
                "Operational note: the backup coolant pump installed in bay 7 carries "
                "the asset tag PMP-4417 and must be tested every 90 days."
            )
    return "\n".join(blocks)


LONG_DOCUMENT = _long_context_document()


BENCHMARK_PROMPTS: List[Dict[str, Any]] = [
    {
        "id": "reasoning_trains",
        "label": "Multi-step arithmetic reasoning",
        "category": "reasoning",
        "prompt": (
            "A train leaves station A at 09:00 travelling at 60 km/h. A second train "
            "leaves station B at 10:00 travelling toward station A at 90 km/h. The "
            "stations are 420 km apart. At what clock time do the trains meet? "
            "Show your working, then give the final answer on its own last line in the "
            "exact form: ANSWER: HH:MM"
        ),
        "expectations": {
            "correctness_patterns": [r"ANSWER:\s*1[23]:2[34]"],
            "correctness_note": "Correct meeting time is 12:24 (trains close 360 km at 150 km/h).",
            "format": "final_line_marker",
            "format_patterns": [r"(?im)^\s*ANSWER:\s*\d{1,2}:\d{2}\s*$"],
            "requirements": ["show working", "final answer line"],
            "min_words": 30,
        },
    },
    {
        "id": "coding_dedupe",
        "label": "Write a Python function",
        "category": "coding",
        "prompt": (
            "Write a Python function named `dedupe_preserve_order` that takes a list and "
            "returns a new list with duplicates removed while preserving first-seen order. "
            "It must handle unhashable elements without raising. Return only a single "
            "Python code block, no explanation."
        ),
        "expectations": {
            "format": "code",
            "format_patterns": [r"```(?:python)?[\s\S]+?```|def\s+dedupe_preserve_order"],
            "correctness_patterns": [r"def\s+dedupe_preserve_order\s*\("],
            "correctness_note": "Checks that the requested function signature is defined.",
            "python_symbol": "dedupe_preserve_order",
            "requirements": ["function defined", "code block only"],
            "forbidden_patterns": [r"(?i)^\s*(sure|certainly|here'?s)\b"],
        },
    },
    {
        "id": "summarization_release",
        "label": "Constrained summarisation",
        "category": "summarization",
        "prompt": (
            "Summarise the following release note in exactly three bullet points, each "
            "under 20 words. Do not add a preamble or a closing sentence.\n\n"
            "Release 4.2 introduces incremental indexing, cutting cold-start indexing time "
            "on large repositories from roughly 40 minutes to under 6 minutes. The query "
            "planner now caches column statistics between sessions, which improves repeat "
            "query latency but increases baseline memory use by about 180 MB. A long-standing "
            "bug that dropped the final row of CSV exports without a trailing newline has been "
            "fixed. The legacy /v1/search endpoint is deprecated and will be removed in 5.0; "
            "callers should migrate to /v2/query."
        ),
        "expectations": {
            "format": "bullets",
            "format_patterns": [r"(?m)^\s*[-*\u2022]\s+\S"],
            "required_bullets": 3,
            "max_words_per_bullet": 20,
            "requirements": ["exactly three bullets", "under 20 words each", "no preamble"],
            "keywords_any": ["index", "cache", "csv", "deprecat", "/v2", "memory"],
            "forbidden_patterns": [r"(?i)^\s*(here'?s|sure|summary:)"],
        },
    },
    {
        "id": "factual_qa_units",
        "label": "Factual question answering",
        "category": "factual_qa",
        "prompt": (
            "Answer in one sentence: what is the SI base unit of electric current, and "
            "what symbol represents it?"
        ),
        "expectations": {
            "correctness_patterns": [r"(?i)\bamp(ere)?s?\b"],
            "correctness_secondary": [r"(?i)\bA\b|symbol\s+A"],
            "correctness_note": "Expected answer: the ampere, symbol A.",
            "format": "prose",
            "max_words": 60,
            "requirements": ["one sentence", "names unit and symbol"],
        },
    },
    {
        "id": "creative_lighthouse",
        "label": "Creative writing",
        "category": "creative",
        "prompt": (
            "Write a 120-word scene about a lighthouse keeper who finds a message in a "
            "bottle. Write in the past tense and do not use the word 'storm'."
        ),
        "expectations": {
            "format": "prose",
            "min_words": 70,
            "max_words": 200,
            "forbidden_words": ["storm"],
            "requirements": ["past tense", "avoids the word 'storm'", "about 120 words"],
            "correctness_scorable": False,
            "correctness_note": "Open-ended creative task: correctness is not objectively scorable.",
        },
    },
    {
        "id": "instruction_following_strict",
        "label": "Strict instruction following",
        "category": "instruction_following",
        "prompt": (
            "Follow these instructions exactly:\n"
            "1. Reply with exactly four lines.\n"
            "2. Each line must start with the word 'Step' followed by its number.\n"
            "3. Describe how to safely jump-start a car, one action per line.\n"
            "4. Do not include any other text, headings, or closing remarks."
        ),
        "expectations": {
            "format": "lines",
            "format_patterns": [r"(?m)^\s*Step\s*1\b"],
            "required_lines": 4,
            "line_prefix": "Step",
            "requirements": ["exactly four lines", "each starts with Step N", "no extra text"],
            "keywords_any": ["cable", "clamp", "terminal", "battery", "engine", "ground"],
        },
    },
    {
        "id": "json_structured_invoice",
        "label": "Structured JSON generation",
        "category": "json_structured",
        "prompt": (
            "Extract the fields from this text and return ONLY valid JSON with the keys "
            "\"invoice_id\" (string), \"total\" (number), \"currency\" (string), and "
            "\"line_items\" (array of objects with \"name\" and \"amount\"). No markdown, "
            "no commentary.\n\n"
            "Invoice INV-2094 issued to Harbour Supplies. Two line items: dock cleats at "
            "245.50 and mooring rope at 89.25. Total due 334.75 EUR."
        ),
        "expectations": {
            "format": "json",
            "json_required_keys": ["invoice_id", "total", "currency", "line_items"],
            "correctness_patterns": [r"INV-2094"],
            "correctness_secondary": [r"334\.75", r"(?i)EUR"],
            "correctness_note": "Expected invoice_id INV-2094, total 334.75, currency EUR.",
            "requirements": ["valid JSON", "all four keys", "no markdown fence"],
            "supports_json_mode": True,
        },
    },
    {
        "id": "long_context_lookup",
        "label": "Long-context retrieval",
        "category": "long_context",
        "prompt": (
            "Read the operations log below and answer one question.\n\n"
            "=== LOG START ===\n" + LONG_DOCUMENT + "\n=== LOG END ===\n\n"
            "Question: what is the asset tag of the backup coolant pump in bay 7, and how "
            "often must it be tested? Answer in one short sentence."
        ),
        "expectations": {
            "correctness_patterns": [r"PMP-4417"],
            "correctness_secondary": [r"90"],
            "correctness_note": "The log states asset tag PMP-4417, tested every 90 days.",
            "format": "prose",
            "max_words": 60,
            "requirements": ["finds the asset tag", "states the 90-day interval"],
            "long_context": True,
        },
    },
]


def get_prompt(prompt_id: str) -> Optional[Dict[str, Any]]:
    for item in BENCHMARK_PROMPTS:
        if item["id"] == prompt_id:
            return item
    return None


def list_prompts() -> List[Dict[str, Any]]:
    """Catalogue for the UI (prompt text truncated for the long-context entry)."""
    out = []
    for item in BENCHMARK_PROMPTS:
        preview = item["prompt"]
        if len(preview) > 600:
            preview = preview[:400] + "\n\n[... document truncated in this preview ...]"
        out.append({
            "id": item["id"],
            "label": item["label"],
            "category": item["category"],
            "preview": preview,
            "prompt": item["prompt"],
            "characters": len(item["prompt"]),
            "requirements": item["expectations"].get("requirements", []),
            "correctness_scorable": item["expectations"].get("correctness_scorable", True)
                                    and bool(item["expectations"].get("correctness_patterns")),
        })
    return out


CATEGORIES = sorted({p["category"] for p in BENCHMARK_PROMPTS})
