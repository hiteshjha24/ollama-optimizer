"""Quality evaluation.

The default evaluator is **deterministic and heuristic**. It checks observable
properties of the produced text: did it match the requested format, did it obey
the explicit constraints in the prompt, does it contain the verifiable answer
where one exists, is it free of degenerate repetition, and so on.

It does not measure hidden reasoning, and it is not a substitute for human
judgement. Every criterion returns the evidence behind its score so a user can
disagree with it, and criteria that cannot be checked for a given prompt are
reported as *not scorable* instead of being given a made-up number.

An optional second evaluator uses another Ollama model as a judge. Those scores
are always labelled as estimates produced by a language model.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .logging_setup import get_logger

log = get_logger("evaluation")

MAX_SCORE = 10.0

# Relative weights of the quality criteria. Criteria that are not scorable for a
# given prompt are dropped and the remaining weights are renormalised.
EVALUATOR_ID = "heuristic-v1"

CRITERION_WEIGHTS = {
    "correctness": 0.28,
    "instruction_following": 0.22,
    "completeness": 0.16,
    "format_compliance": 0.16,
    "relevance": 0.10,
    "coherence": 0.08,
}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "for", "with", "as", "by", "at", "from", "it", "its",
    "you", "your", "i", "we", "they", "he", "she", "do", "does", "did", "not",
    "no", "yes", "can", "will", "would", "should", "could", "may", "might",
    "must", "have", "has", "had", "about", "into", "over", "under", "each",
    "any", "all", "one", "two", "three", "four", "five", "please", "write",
    "give", "answer", "response", "using", "use", "there", "their", "what",
    "which", "who", "how", "when", "where", "why", "so", "up", "out", "more",
}

_WORD_RE = re.compile(r"[A-Za-z0-9_'-]+")


# --------------------------------------------------------------------------
# small text utilities
# --------------------------------------------------------------------------

def words(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def word_count(text: str) -> int:
    return len(words(text))


def content_tokens(text: str) -> List[str]:
    return [w.lower() for w in words(text) if w.lower() not in _STOPWORDS and len(w) > 2]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def extract_json_blob(text: str) -> Tuple[Optional[Any], bool, str]:
    """Find a JSON value in the output.

    Returns (parsed, was_fenced, note). ``parsed`` is None when nothing parses.
    """
    if not text:
        return None, False, "empty output"
    stripped = text.strip()
    fenced = False
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
    candidate = stripped
    if fence:
        candidate = fence.group(1).strip()
        fenced = True
    try:
        return json.loads(candidate), fenced, "parsed directly"
    except json.JSONDecodeError:
        pass
    # fall back to the largest brace/bracket span
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start:end + 1]), fenced, "parsed from embedded span"
            except json.JSONDecodeError:
                continue
    return None, fenced, "no parsable JSON found"


def repetition_ratio(text: str, n: int = 4) -> Optional[float]:
    """Share of repeated n-grams (a proxy for degenerate looping)."""
    toks = [w.lower() for w in words(text)]
    if len(toks) < n * 3:
        return None
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return None
    return 1.0 - (len(set(grams)) / len(grams))


def _clamp(value: float) -> float:
    return max(0.0, min(MAX_SCORE, value))


@dataclass
class Criterion:
    name: str
    score: Optional[float]
    scorable: bool
    basis: str
    evidence: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["max"] = MAX_SCORE
        return d


# --------------------------------------------------------------------------
# expectation inference for user-supplied prompts
# --------------------------------------------------------------------------

def infer_expectations(prompt: str, preset: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Derive checkable constraints from an arbitrary prompt.

    A preset (predefined benchmark prompt) always wins; inference only fills in
    what the user's own prompt states explicitly.
    """
    if preset:
        exp = dict(preset)
        exp.setdefault("source", "predefined benchmark prompt")
        return exp

    text = prompt or ""
    lower = text.lower()
    exp: Dict[str, Any] = {
        "source": "inferred from the wording of your prompt",
        "requirements": [],
        "correctness_scorable": False,
        "correctness_note": (
            "No verifiable answer key exists for a custom prompt, so correctness is not "
            "scored deterministically."
        ),
    }

    if re.search(r"\bjson\b", lower):
        exp["format"] = "json"
        exp["requirements"].append("valid JSON output")
        keys = re.findall(r'"([A-Za-z0-9_]+)"', text)
        if keys:
            exp["json_required_keys"] = sorted(set(keys))[:12]
    elif re.search(r"\bbullet(s| points?)?\b|\blist\b", lower):
        exp["format"] = "bullets"
        exp["format_patterns"] = [r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+\S"]
        exp["requirements"].append("bulleted or numbered list")
    elif re.search(r"\bcode\b|\bfunction\b|\bscript\b|\bpython\b|\bjavascript\b", lower):
        exp["format"] = "code"
        exp["format_patterns"] = [r"```[\s\S]+?```|(?m)^\s*(?:def|class|function|const|import)\s"]
        exp["requirements"].append("contains code")
    elif re.search(r"\btable\b|\bmarkdown table\b", lower):
        exp["format"] = "table"
        exp["format_patterns"] = [r"(?m)^\s*\|.+\|\s*$"]
        exp["requirements"].append("markdown table")
    else:
        exp["format"] = "prose"

    m = re.search(r"\b(?:in|under|within|at most|no more than|max(?:imum)? of)\s+(\d{1,4})\s*words\b", lower)
    if m:
        exp["max_words"] = int(m.group(1)) * 1.35  # tolerance band
        exp["stated_word_limit"] = int(m.group(1))
        exp["requirements"].append(f"about {m.group(1)} words or fewer")
    m = re.search(r"\b(?:at least|minimum of|no fewer than)\s+(\d{1,4})\s*words\b", lower)
    if m:
        exp["min_words"] = int(m.group(1)) * 0.75
        exp["requirements"].append(f"at least {m.group(1)} words")

    m = re.search(r"\bexactly\s+(\w+)\s+(lines|bullet points|bullets|sentences|items|paragraphs)\b", lower)
    if m:
        count = _word_to_int(m.group(1))
        unit = m.group(2)
        if count:
            if "line" in unit:
                exp["required_lines"] = count
            elif "bullet" in unit or "item" in unit:
                exp["required_bullets"] = count
            elif "sentence" in unit:
                exp["required_sentences"] = count
            elif "paragraph" in unit:
                exp["required_paragraphs"] = count
            exp["requirements"].append(f"exactly {count} {unit}")

    forbidden = re.findall(r"do not use the word ['\"]?([A-Za-z-]+)['\"]?", lower)
    if forbidden:
        exp["forbidden_words"] = forbidden
        exp["requirements"].extend(f"avoids the word '{w}'" for w in forbidden)

    if re.search(r"\bonly\b.*\b(json|code|answer|list)\b|no (commentary|explanation|preamble)", lower):
        exp["no_preamble"] = True
        exp["requirements"].append("no preamble or commentary")

    numbered = re.findall(r"(?m)^\s*(\d)[.)]\s+(.{4,120})$", text)
    if len(numbered) >= 2:
        exp["numbered_instructions"] = [n[1].strip() for n in numbered]
        exp["requirements"].append(f"addresses all {len(numbered)} numbered instructions")

    if not exp["requirements"]:
        exp["requirements"].append("answers the prompt (no explicit constraints detected)")
    return exp


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _word_to_int(token: str) -> Optional[int]:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUM_WORDS.get(token)


# --------------------------------------------------------------------------
# individual criteria
# --------------------------------------------------------------------------

def _score_correctness(output: str, exp: Dict[str, Any]) -> Criterion:
    patterns = exp.get("correctness_patterns") or []
    if exp.get("correctness_scorable") is False or not patterns:
        return Criterion(
            "correctness", None, False,
            "no verifiable answer key for this prompt",
            exp.get("correctness_note", "")
            or "Correctness requires a known answer; none is defined for this prompt.",
        )
    primary_hits = [p for p in patterns if re.search(p, output or "")]
    secondary = exp.get("correctness_secondary") or []
    secondary_hits = [p for p in secondary if re.search(p, output or "")]

    primary_ratio = len(primary_hits) / len(patterns)
    score = primary_ratio * 8.0
    if secondary:
        score += (len(secondary_hits) / len(secondary)) * 2.0
    else:
        score += 2.0 * primary_ratio
    evidence = (f"matched {len(primary_hits)}/{len(patterns)} required answer patterns"
                + (f", {len(secondary_hits)}/{len(secondary)} supporting patterns"
                   if secondary else ""))
    if exp.get("correctness_note"):
        evidence += f". {exp['correctness_note']}"
    return Criterion("correctness", round(_clamp(score), 2), True,
                     "regex match against the known answer for this benchmark prompt",
                     evidence)


def _score_format(output: str, exp: Dict[str, Any]) -> Criterion:
    fmt = exp.get("format", "prose")
    text = output or ""
    notes: List[str] = []

    if fmt == "json":
        parsed, fenced, note = extract_json_blob(text)
        if parsed is None:
            return Criterion("format_compliance", 0.0, True,
                             "requested JSON could not be parsed", note)
        score = 10.0
        notes.append(f"JSON {note}")
        if fenced:
            score -= 2.0
            notes.append("wrapped in a markdown fence although none was requested")
        required = exp.get("json_required_keys") or []
        if required and isinstance(parsed, dict):
            missing = [k for k in required if k not in parsed]
            if missing:
                score -= min(6.0, 6.0 * len(missing) / len(required))
                notes.append(f"missing keys: {', '.join(missing)}")
            else:
                notes.append(f"all {len(required)} required keys present")
        elif required:
            score -= 4.0
            notes.append("JSON parsed but is not an object with the requested keys")
        return Criterion("format_compliance", round(_clamp(score), 2), True,
                         "JSON parse + required-key check", "; ".join(notes))

    if fmt == "bullets":
        bullets = re.findall(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+\S.*$", text)
        required = exp.get("required_bullets")
        if not bullets:
            return Criterion("format_compliance", 0.0, True,
                             "bulleted list requested", "no bullet lines found")
        score = 8.0
        notes.append(f"{len(bullets)} bullet lines")
        if required:
            delta = abs(len(bullets) - required)
            score = 10.0 - min(8.0, delta * 3.0)
            notes.append(f"{required} required")
        else:
            score = 9.0
        limit = exp.get("max_words_per_bullet")
        if limit:
            over = [b for b in bullets if word_count(b) > limit]
            if over:
                score -= min(4.0, len(over))
                notes.append(f"{len(over)} bullet(s) exceed {limit} words")
        return Criterion("format_compliance", round(_clamp(score), 2), True,
                         "bullet structure check", "; ".join(notes))

    if fmt == "code":
        fenced = re.search(r"```[\s\S]+?```", text)
        symbol = exp.get("python_symbol")
        score = 4.0
        if fenced:
            score += 4.0
            notes.append("code block present")
        elif re.search(r"(?m)^\s*(?:def|class|function|const|import)\s", text):
            score += 2.0
            notes.append("code-like lines present but not fenced")
        else:
            notes.append("no code block detected")
        if symbol and re.search(rf"\b{re.escape(symbol)}\b", text):
            score += 2.0
            notes.append(f"defines `{symbol}`")
        elif symbol:
            notes.append(f"`{symbol}` not found")
        return Criterion("format_compliance", round(_clamp(score), 2), True,
                         "code block / symbol detection", "; ".join(notes))

    if fmt == "lines":
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        required = exp.get("required_lines")
        prefix = exp.get("line_prefix")
        score = 10.0
        notes.append(f"{len(lines)} non-empty lines")
        if required:
            score -= min(8.0, abs(len(lines) - required) * 3.0)
            notes.append(f"{required} required")
        if prefix:
            good = [ln for ln in lines if ln.strip().lower().startswith(prefix.lower())]
            ratio = len(good) / len(lines) if lines else 0
            score -= (1 - ratio) * 4.0
            notes.append(f"{len(good)}/{len(lines)} lines start with '{prefix}'")
        return Criterion("format_compliance", round(_clamp(score), 2), True,
                         "line count / prefix check", "; ".join(notes))

    if fmt in ("table",):
        rows = re.findall(r"(?m)^\s*\|.+\|\s*$", text)
        score = 10.0 if len(rows) >= 2 else 2.0
        return Criterion("format_compliance", score, True, "markdown table detection",
                         f"{len(rows)} table rows found")

    if fmt == "final_line_marker":
        patterns = exp.get("format_patterns") or []
        hit = any(re.search(p, text) for p in patterns)
        return Criterion("format_compliance", 10.0 if hit else 3.0, True,
                         "required answer marker line",
                         "answer marker found" if hit else "required marker line missing")

    # prose: mild structural expectations only
    if not text.strip():
        return Criterion("format_compliance", 0.0, True, "prose expected", "empty output")
    score = 9.0
    if exp.get("no_preamble") and re.match(r"(?i)^\s*(sure|certainly|of course|here'?s|here is)\b", text):
        score -= 4.0
        notes.append("starts with a preamble although none was wanted")
    return Criterion("format_compliance", round(_clamp(score), 2), True,
                     "free-form prose expected", "; ".join(notes) or "no format violations detected")


def _score_instruction_following(output: str, exp: Dict[str, Any],
                                 metrics: Dict[str, Any]) -> Criterion:
    text = output or ""
    checks: List[Tuple[str, bool]] = []

    wc = word_count(text)
    if exp.get("max_words"):
        checks.append((f"<= {int(exp['max_words'])} words (measured {wc})",
                       wc <= float(exp["max_words"])))
    if exp.get("min_words"):
        checks.append((f">= {int(exp['min_words'])} words (measured {wc})",
                       wc >= float(exp["min_words"])))
    for word in exp.get("forbidden_words", []) or []:
        checks.append((f"avoids '{word}'",
                       not re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)))
    for pattern in exp.get("forbidden_patterns", []) or []:
        checks.append(("avoids forbidden pattern", not re.search(pattern, text)))
    if exp.get("required_lines"):
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        checks.append((f"exactly {exp['required_lines']} lines (got {len(lines)})",
                       len(lines) == exp["required_lines"]))
    if exp.get("required_bullets"):
        bullets = re.findall(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+\S", text)
        checks.append((f"exactly {exp['required_bullets']} bullets (got {len(bullets)})",
                       len(bullets) == exp["required_bullets"]))
    if exp.get("required_sentences"):
        sentences = [s for s in re.split(r"[.!?]+\s", text.strip()) if s.strip()]
        checks.append((f"exactly {exp['required_sentences']} sentences (got {len(sentences)})",
                       len(sentences) == exp["required_sentences"]))
    if exp.get("no_preamble"):
        checks.append(("no conversational preamble",
                       not re.match(r"(?i)^\s*(sure|certainly|of course|here'?s|here is|okay)\b", text)))
    keywords_any = exp.get("keywords_any") or []
    if keywords_any:
        hits = [k for k in keywords_any if k.lower() in text.lower()]
        checks.append((f"mentions expected topic terms ({len(hits)}/{len(keywords_any)})",
                       bool(hits)))
    if metrics.get("truncated"):
        checks.append(("finished within the token budget (not truncated)", False))

    if not checks:
        return Criterion("instruction_following", None, False,
                         "no explicit, checkable instructions found in the prompt",
                         "Add constraints such as a word limit or an output format to "
                         "make this measurable.")
    passed = sum(1 for _, ok in checks if ok)
    score = MAX_SCORE * passed / len(checks)
    evidence = "; ".join(f"{'PASS' if ok else 'FAIL'}: {label}" for label, ok in checks)
    return Criterion("instruction_following", round(_clamp(score), 2), True,
                     f"{passed}/{len(checks)} explicit constraints satisfied", evidence)


def _score_completeness(output: str, exp: Dict[str, Any], prompt: str,
                        metrics: Dict[str, Any]) -> Criterion:
    text = (output or "").strip()
    if not text:
        return Criterion("completeness", 0.0, True, "empty output", "no content produced")
    notes: List[str] = []
    score = 8.0

    if metrics.get("truncated"):
        score -= 4.0
        notes.append("output hit the token limit and was cut off")

    instructions = exp.get("numbered_instructions") or []
    if instructions:
        covered = 0
        for instruction in instructions:
            toks = content_tokens(instruction)[:6]
            if toks and sum(1 for t in toks if t in text.lower()) >= max(1, len(toks) // 3):
                covered += 1
        ratio = covered / len(instructions)
        score = 3.0 + 7.0 * ratio
        notes.append(f"addressed {covered}/{len(instructions)} numbered instructions "
                     "(lexical coverage check)")

    stated_limit = exp.get("stated_word_limit") or exp.get("max_words")
    wc = word_count(text)
    if stated_limit and wc < float(stated_limit) * 0.3:
        score -= 2.0
        notes.append(f"only {wc} words against a target of about {int(float(stated_limit))}")
    if wc < 5 and exp.get("format") != "json":
        score -= 3.0
        notes.append("response is extremely short")

    if exp.get("format") == "json":
        parsed, _, _ = extract_json_blob(text)
        required = exp.get("json_required_keys") or []
        if parsed is not None and required and isinstance(parsed, dict):
            present = [k for k in required if k in parsed and parsed.get(k) not in (None, "", [])]
            ratio = len(present) / len(required)
            score = 3.0 + 7.0 * ratio
            notes.append(f"{len(present)}/{len(required)} requested fields populated")

    return Criterion("completeness", round(_clamp(score), 2), True,
                     "coverage of the requested elements",
                     "; ".join(notes) or "no missing elements detected")


def _score_relevance(output: str, prompt: str) -> Criterion:
    out_tokens = content_tokens(output or "")
    prompt_tokens = content_tokens(prompt or "")
    if not out_tokens:
        return Criterion("relevance", 0.0, True, "lexical overlap with the prompt",
                         "empty output")
    if not prompt_tokens:
        return Criterion("relevance", None, False, "prompt has no content words", "")
    # coverage of prompt terms, capped: a good answer need not repeat the prompt
    prompt_set = set(prompt_tokens)
    hits = sum(1 for t in set(out_tokens) if t in prompt_set)
    coverage = hits / max(1, min(len(prompt_set), 40))
    score = _clamp(3.0 + 7.0 * min(1.0, coverage * 1.6))
    return Criterion("relevance", round(score, 2), True,
                     "share of the prompt's content words that reappear in the answer",
                     f"{hits} overlapping content terms; coverage {coverage:.2f} "
                     "(a lexical proxy, not a semantic judgement)")


def _score_coherence(output: str, metrics: Dict[str, Any]) -> Criterion:
    text = (output or "").strip()
    if not text:
        return Criterion("coherence", 0.0, True, "structural checks", "empty output")
    notes: List[str] = []
    score = 9.0

    rep = repetition_ratio(text)
    if rep is not None:
        if rep > 0.45:
            score -= 6.0
            notes.append(f"high 4-gram repetition ({rep:.0%}) suggests looping")
        elif rep > 0.25:
            score -= 2.5
            notes.append(f"moderate 4-gram repetition ({rep:.0%})")
        else:
            notes.append(f"4-gram repetition {rep:.0%}")
    else:
        notes.append("too short to measure repetition")

    if metrics.get("truncated"):
        score -= 2.0
        notes.append("cut off mid-generation at the token limit")
    elif not re.search(r"[.!?)\]}`\"']\s*$", text):
        score -= 1.0
        notes.append("does not end on a terminal character")

    toks = words(text)
    if toks:
        unique_ratio = len(set(w.lower() for w in toks)) / len(toks)
        if unique_ratio < 0.25 and len(toks) > 40:
            score -= 3.0
            notes.append(f"low lexical variety ({unique_ratio:.0%} unique tokens)")

    if re.search(r"(.{12,}?)\1{2,}", text):
        score -= 3.0
        notes.append("a long substring repeats three or more times")

    return Criterion("coherence", round(_clamp(score), 2), True,
                     "repetition, truncation and lexical-variety checks",
                     "; ".join(notes))


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def evaluate_output(prompt: str, output: str, expectations: Dict[str, Any],
                    metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score one generated output. Deterministic: same input -> same score."""
    metrics = metrics or {}
    if not (output or "").strip():
        # An empty response satisfies nothing; do not award partial credit.
        empty = [Criterion(name, 0.0, True, "empty response: no output was produced")
                 for name in CRITERION_WEIGHTS]
        return {
            "evaluator": EVALUATOR_ID,
            "quality_score": 0.0,
            "quality_basis": "empty response - every criterion scores 0",
            "criteria": {c.name: c.as_dict() for c in empty},
            "weights": dict(CRITERION_WEIGHTS),
            "not_scorable": [],
            "word_count": 0,
            "limitations": "The model returned no text for this run.",
        }
    criteria = [
        _score_correctness(output, expectations),
        _score_instruction_following(output, expectations, metrics),
        _score_completeness(output, expectations, prompt, metrics),
        _score_format(output, expectations),
        _score_relevance(output, prompt),
        _score_coherence(output, metrics),
    ]
    by_name = {c.name: c for c in criteria}

    used_weight = 0.0
    total = 0.0
    for name, weight in CRITERION_WEIGHTS.items():
        crit = by_name.get(name)
        if crit and crit.scorable and crit.score is not None:
            total += crit.score * weight
            used_weight += weight
    quality = round(total / used_weight, 3) if used_weight else None

    return {
        "evaluator": EVALUATOR_ID,
        "quality_score": quality,
        "quality_basis": (
            f"weighted mean of {sum(1 for c in criteria if c.scorable)} scorable criteria "
            f"(weights renormalised over {used_weight:.2f} of 1.00)"
        ),
        "criteria": {c.name: c.as_dict() for c in criteria},
        "weights": {k: v for k, v in CRITERION_WEIGHTS.items()
                    if by_name.get(k) and by_name[k].scorable},
        "not_scorable": [c.name for c in criteria if not c.scorable],
        "word_count": word_count(output or ""),
        "limitations": (
            "Heuristic scores check observable output properties only (format, explicit "
            "constraints, answer patterns, repetition). They do not assess factual "
            "accuracy for open-ended prompts or the model's internal reasoning."
        ),
    }


def consistency_across_runs(outputs: Sequence[str],
                            quality_scores: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Score how reproducible a configuration is across its repeated runs."""
    valid_outputs = [o for o in outputs if o]
    scores = [s for s in quality_scores if isinstance(s, (int, float))]

    if len(valid_outputs) < 2:
        return {"score": None, "scorable": False,
                "basis": "at least two successful runs are needed",
                "text_similarity": None, "quality_stdev": None,
                "identical_outputs": len(valid_outputs) == 1}

    token_sets = [content_tokens(o) for o in valid_outputs]
    sims = [jaccard(token_sets[i], token_sets[j])
            for i in range(len(token_sets)) for j in range(i + 1, len(token_sets))]
    mean_sim = statistics.fmean(sims) if sims else 0.0

    quality_stdev = statistics.stdev(scores) if len(scores) >= 2 else 0.0
    # 0 stdev -> full marks; 2.0 stdev on a 0-10 scale -> zero marks
    stability = max(0.0, 1.0 - (quality_stdev / 2.0))

    identical = len(set(valid_outputs)) == 1
    score = _clamp(10.0 * (0.55 * mean_sim + 0.45 * stability))

    return {
        "score": round(score, 2),
        "scorable": True,
        "basis": ("55% mean pairwise content-word similarity + 45% stability of the "
                  "quality score across runs"),
        "text_similarity": round(mean_sim, 4),
        "quality_stdev": round(quality_stdev, 4),
        "identical_outputs": identical,
        "n": len(valid_outputs),
        "note": ("Identical outputs across runs usually mean a fixed seed or "
                 "temperature 0." if identical else None),
    }


# --------------------------------------------------------------------------
# optional LLM judge
# --------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a strict evaluation assistant. You will be given a task prompt and a "
    "candidate response. Score the response from 0 to 10 on each criterion. "
    "Reply with ONLY a JSON object with the keys correctness, relevance, "
    "instruction_following, completeness, coherence, and rationale. The first five "
    "must be numbers; rationale must be one short sentence."
)

JUDGE_KEYS = ["correctness", "relevance", "instruction_following", "completeness", "coherence"]


def llm_judge(service: Any, model: str, prompt: str, output: str,
              timeout: float = 90.0) -> Dict[str, Any]:
    """Ask a second Ollama model to score an output.

    Returns a dict that is always labelled as an estimate, and reports failure
    rather than substituting numbers when the judge misbehaves.
    """
    user = (
        f"TASK PROMPT:\n{prompt[:4000]}\n\n"
        f"CANDIDATE RESPONSE:\n{(output or '')[:6000]}\n\n"
        "Return only the JSON object."
    )
    try:
        result = service.chat(
            model,
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": user}],
            options={"temperature": 0.0, "seed": 42, "num_predict": 300},
            timeout=timeout, fmt="json",
        )
    except Exception as exc:  # any Ollama failure
        return {"available": False, "reason": f"judge call failed: {exc}",
                "evaluator": f"llm-judge:{model}"}

    parsed, _, note = extract_json_blob(result.text)
    if not isinstance(parsed, dict):
        return {"available": False, "reason": f"judge did not return JSON ({note})",
                "evaluator": f"llm-judge:{model}", "raw": (result.text or "")[:300]}

    scores: Dict[str, Optional[float]] = {}
    for key in JUDGE_KEYS:
        value = parsed.get(key)
        if isinstance(value, (int, float)) and 0 <= float(value) <= 10:
            scores[key] = round(float(value), 2)
        else:
            scores[key] = None
    usable = [v for v in scores.values() if v is not None]
    return {
        "available": bool(usable),
        "evaluator": f"llm-judge:{model}",
        "scores": scores,
        "mean": round(statistics.fmean(usable), 3) if usable else None,
        "rationale": str(parsed.get("rationale", ""))[:400],
        "label": "ESTIMATE - generated by a language model, not an objective measurement",
        "judge_latency_seconds": round(result.wall_seconds, 3),
    }
