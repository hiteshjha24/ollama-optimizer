"""Context optimization.

Two distinct things are tested here and reported separately:

* **Context window size** (``num_ctx``) - only within the length the model
  actually declares. Changing ``num_ctx`` forces Ollama to reload the model, so
  these configurations also expose load cost.
* **Context content** - trimming filler and moving the question ahead of a long
  document. Only attempted when the prompt is long enough for it to matter.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import Applicability, Optimization, OptimizationContext, RunConfig

# Rough token estimate for English text. Used only to avoid proposing a context
# window smaller than the prompt; it is never reported as a token measurement.
CHARS_PER_TOKEN = 3.7
LONG_PROMPT_CHARS = 1200


def estimate_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


class ContextOptimization(Optimization):
    id = "context"
    name = "Context optimization"
    category = "context"
    description = (
        "Tests context-window sizes the model actually supports, and - for long "
        "prompts - whether trimming or reordering the context changes quality, "
        "latency or memory behaviour."
    )
    parameters = ["num_ctx", "prompt layout"]
    fine_budget = 1

    def applicability(self, ctx: OptimizationContext) -> Applicability:
        declared = ctx.capabilities.get("context_length")
        long_prompt = len(ctx.prompt or "") >= LONG_PROMPT_CHARS
        if not declared and not long_prompt:
            return Applicability.skip(
                "Ollama did not report a context length for this model and the prompt "
                "is short, so there is nothing safe to vary. Setting num_ctx blind "
                "could exceed what the model supports."
            )
        if not declared:
            return Applicability.ok(
                "Context length is not reported by /api/show, so only content-level "
                "context changes are tested; num_ctx is left at the runtime default."
            )
        return Applicability.ok(
            f"Model declares a context length of {declared} tokens."
        )

    def generate_configurations(self, ctx: OptimizationContext) -> List[RunConfig]:
        base = ctx.baseline_options
        configs: List[RunConfig] = []
        declared = ctx.capabilities.get("context_length")
        needed = estimate_tokens(ctx.prompt) + int(ctx.max_tokens) + 128

        if declared:
            candidates = sorted({
                c for c in (2048, 4096, 8192, int(declared))
                if needed <= c <= int(declared)
            })
            # keep at most three sizes: smallest workable, a middle one, the maximum
            if len(candidates) > 3:
                candidates = [candidates[0], candidates[len(candidates) // 2], candidates[-1]]
            for size in candidates:
                configs.append(RunConfig(
                    key=f"ctx_num_ctx_{size}",
                    label=f"num_ctx = {size}",
                    optimization_id=self.id,
                    category=self.category,
                    phase="coarse",
                    options={**base, "num_ctx": size},
                    rationale=(
                        f"Sets the context window to {size} tokens (the model declares "
                        f"{declared}; this prompt needs roughly {needed}). A smaller "
                        "window reduces KV-cache memory and can speed up prompt "
                        "evaluation; a larger one costs memory for no benefit when the "
                        "prompt does not need it. Changing num_ctx reloads the model, "
                        "so load time is reported separately."
                    ),
                    tags=["num_ctx", "reload"],
                ))

        prompt = ctx.prompt or ""
        if len(prompt) >= LONG_PROMPT_CHARS:
            trimmed = _trim_context(prompt)
            if trimmed and len(trimmed) < len(prompt) * 0.92:
                configs.append(RunConfig(
                    key="ctx_trimmed",
                    label="Trimmed context",
                    optimization_id=self.id,
                    category=self.category,
                    phase="coarse",
                    options=dict(base),
                    prompt=trimmed,
                    rationale=(
                        f"Removes repeated and boilerplate lines, cutting the prompt "
                        f"from {len(prompt)} to {len(trimmed)} characters. Tests whether "
                        "the removed material was carrying any weight, and how much "
                        "prompt-evaluation time it cost."
                    ),
                    tags=["context_content", "trimmed"],
                ))

            reordered = _question_first(prompt)
            if reordered and reordered != prompt:
                configs.append(RunConfig(
                    key="ctx_question_first",
                    label="Question before document",
                    optimization_id=self.id,
                    category=self.category,
                    phase="coarse",
                    options=dict(base),
                    prompt=reordered,
                    rationale=(
                        "Moves the instruction ahead of the long document so the model "
                        "reads the task before the material. Tests the ordering effect "
                        "on retrieval quality without changing any content."
                    ),
                    tags=["context_content", "reordered"],
                ))
        return configs


def _trim_context(prompt: str) -> Optional[str]:
    """Drop duplicate lines and obvious boilerplate, preserving order."""
    lines = prompt.splitlines()
    if len(lines) < 8:
        return None
    seen: set[str] = set()
    kept: List[str] = []
    for line in lines:
        norm = re.sub(r"\s+", " ", line.strip().lower())
        if not norm:
            if kept and kept[-1] == "":
                continue
            kept.append("")
            continue
        if norm in seen and len(norm) > 25:
            continue
        seen.add(norm)
        kept.append(line)
    trimmed = "\n".join(kept).strip()
    return trimmed or None


def _question_first(prompt: str) -> Optional[str]:
    """Hoist a trailing question/instruction above a long body of text."""
    lines = [ln for ln in prompt.splitlines()]
    if len(lines) < 6:
        return None
    tail = lines[-6:]
    question_idx = None
    for offset, line in enumerate(tail):
        if re.search(r"(?i)^(question|task|instruction)\b|\?\s*$", line.strip()):
            question_idx = len(lines) - 6 + offset
            break
    if question_idx is None:
        return None
    head = lines[question_idx:]
    body = lines[:question_idx]
    question_text = "\n".join(head).strip()
    body_text = "\n".join(body).strip()
    if not question_text or not body_text:
        return None
    return f"{question_text}\n\nUse only the material below to answer.\n\n{body_text}"
