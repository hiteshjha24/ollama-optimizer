"""Prompt-level and system-prompt optimizations.

The user's original prompt is never discarded: every variant is stored in full
alongside the original so the report can show exactly what changed.

These experiments measure **observable output quality only**. They do not, and
cannot, measure hidden reasoning; a variant that asks the model to work step by
step is scored on the text it produces, nothing more.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import (Applicability, ConfigOutcome, Optimization, OptimizationContext,
                   RunConfig, best_by_quality)


def _format_clause(expectations: Dict[str, Any]) -> str:
    fmt = (expectations or {}).get("format", "prose")
    if fmt == "json":
        keys = expectations.get("json_required_keys") or []
        if keys:
            return ("Output: a single valid JSON object with exactly these keys: "
                    + ", ".join(keys) + ". No markdown fence, no commentary.")
        return "Output: a single valid JSON object. No markdown fence, no commentary."
    if fmt == "bullets":
        count = expectations.get("required_bullets")
        return (f"Output: {count} bullet points and nothing else."
                if count else "Output: a bulleted list and nothing else.")
    if fmt == "code":
        return "Output: one code block containing the complete implementation, nothing else."
    if fmt == "lines":
        count = expectations.get("required_lines")
        return (f"Output: exactly {count} lines, nothing before or after."
                if count else "Output: one item per line, nothing else.")
    if fmt == "final_line_marker":
        return "Output: your working first, then the final answer alone on the last line."
    limit = expectations.get("stated_word_limit")
    return (f"Output: prose of about {limit} words."
            if limit else "Output: a direct prose answer with no preamble.")


def _requirements_clause(expectations: Dict[str, Any]) -> str:
    reqs = (expectations or {}).get("requirements") or []
    if not reqs:
        return ""
    lines = "\n".join(f"- {r}" for r in reqs[:8])
    return f"Requirements:\n{lines}"


def _condense(prompt: str) -> str:
    """Strip politeness and filler without changing the task."""
    text = prompt
    text = re.sub(r"(?i)\b(please|kindly|if you could|i would like you to|i want you to)\b,?\s*",
                  "", text)
    text = re.sub(r"(?i)\b(make sure (that )?you|be sure to|remember to)\b\s*", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class PromptOptimization(Optimization):
    id = "prompt_optimization"
    name = "Prompt optimization"
    category = "prompt"
    description = (
        "Rewrites the user's prompt into structured variants (explicit task framing, "
        "output-format specification, constraints, step decomposition, condensed form) "
        "and measures whether the rewrite changes observable output quality."
    )
    parameters = ["prompt text"]
    fine_budget = 2

    def applicability(self, ctx: OptimizationContext) -> Applicability:
        if len((ctx.prompt or "").strip()) < 12:
            return Applicability.skip(
                "The prompt is too short to rewrite meaningfully (under 12 characters)."
            )
        return Applicability.ok("Prompt rewriting applies to any text prompt.")

    def generate_configurations(self, ctx: OptimizationContext) -> List[RunConfig]:
        original = ctx.prompt.strip()
        exp = ctx.expectations or {}
        base = ctx.baseline_options
        fmt_clause = _format_clause(exp)
        req_clause = _requirements_clause(exp)
        variants: List[tuple[str, str, str, str]] = []

        # 1. explicit task framing
        variants.append((
            "structured",
            "Structured task framing",
            f"Task:\n{original}\n\n{fmt_clause}",
            "Separates the task from the output contract with explicit headings. Tests "
            "whether labelling the sections helps the model keep them apart.",
        ))

        # 2. requirements checklist
        if req_clause:
            variants.append((
                "checklist",
                "Explicit requirements checklist",
                f"Task:\n{original}\n\n{req_clause}\n\n{fmt_clause}",
                "Restates the detected constraints as an explicit checklist. Tests "
                "whether enumerating requirements improves instruction following.",
            ))

        # 3. role + context
        role = _role_for(ctx.prompt_category, exp)
        variants.append((
            "role",
            "Role and context framing",
            f"{role}\n\n{original}\n\n{fmt_clause}",
            "Prefixes a short role statement. Tests the common claim that assigning a "
            "role improves domain-appropriate output; the claim is measured, not assumed.",
        ))

        # 4. step decomposition (only where a process is plausible)
        if _benefits_from_steps(ctx.prompt_category, exp):
            variants.append((
                "decomposed",
                "Step decomposition",
                (f"{original}\n\nWork through the task in order: first identify what is "
                 f"being asked, then do the work, then produce the final output.\n\n{fmt_clause}"),
                "Asks for an ordered approach. Only the visible output is scored - this "
                "does not measure hidden reasoning.",
            ))

        # 5. condensed
        condensed = _condense(original)
        if condensed and condensed != original and len(condensed) < len(original) * 0.95:
            variants.append((
                "condensed",
                "Condensed prompt",
                condensed,
                "Removes filler and politeness while preserving the task. Tests whether "
                "a shorter prompt is processed faster without losing quality.",
            ))

        # 6. output contract only (minimal addition)
        variants.append((
            "format_only",
            "Output-format specification only",
            f"{original}\n\n{fmt_clause}",
            "Adds only the output contract, changing nothing else. Isolates the effect "
            "of format instructions from the other rewrites.",
        ))

        configs = []
        for key, label, text, rationale in variants:
            configs.append(RunConfig(
                key=f"prompt_{key}",
                label=label,
                optimization_id=self.id,
                category=self.category,
                phase="coarse",
                options=dict(base),
                prompt=text,
                rationale=rationale,
                tags=["prompt_variant", key],
            ))
        return configs

    def refine(self, ctx: OptimizationContext,
               results: List[ConfigOutcome]) -> List[RunConfig]:
        """Combine the winning prompt rewrite with a low-temperature setting."""
        best = best_by_quality(results)
        if best is None or best.config.prompt is None:
            return []
        return [RunConfig(
            key=f"{best.config.key}_lowtemp",
            label=f"{best.config.label} + temperature 0.2",
            optimization_id=self.id,
            category=self.category,
            phase="fine",
            options={**ctx.baseline_options, "temperature": 0.2},
            prompt=best.config.prompt,
            rationale=(
                f"'{best.config.label}' scored highest among the prompt rewrites "
                f"(quality {best.quality:.2f}). It is retested with a lower temperature "
                "to check whether prompt structure and sampling settings compound."
            ),
            tags=["fine", "prompt_variant"],
        )]


def _role_for(category: Optional[str], exp: Dict[str, Any]) -> str:
    mapping = {
        "coding": "You are an experienced software engineer writing production code.",
        "reasoning": "You are a careful analyst who checks each step of a calculation.",
        "summarization": "You are an editor who writes tight, factual summaries.",
        "factual_qa": "You are a precise reference assistant.",
        "creative": "You are a fiction writer with a spare, concrete style.",
        "json_structured": "You are a data-extraction service that emits machine-readable output.",
        "long_context": "You are a document analyst who cites only what the source states.",
        "instruction_following": "You are an assistant that follows formatting instructions exactly.",
    }
    if category and category in mapping:
        return mapping[category]
    if (exp or {}).get("format") == "json":
        return mapping["json_structured"]
    return "You are a precise, helpful assistant."


def _benefits_from_steps(category: Optional[str], exp: Dict[str, Any]) -> bool:
    if category in {"reasoning", "coding", "long_context", "json_structured"}:
        return True
    return bool((exp or {}).get("numbered_instructions"))


SYSTEM_PROMPT_VARIANTS = [
    ("minimal", "Minimal system instruction", "You are a helpful assistant.",
     "The shortest plausible system prompt. Acts as the control for this category: "
     "if richer system prompts do not beat it, they are not worth the tokens."),
    ("accuracy", "Accuracy-focused", (
        "You are a precise assistant. Answer only what was asked. If you are not "
        "certain of a fact, say so rather than guessing."),
     "Tests whether an accuracy directive changes measurable output quality or simply "
     "makes the model more hedging."),
    ("concise", "Concise-response", (
        "You are a concise assistant. Give the answer directly, with no preamble, "
        "restatement of the question, or closing summary."),
     "Tests whether suppressing preamble lowers latency and token count without "
     "losing content."),
    ("format", "Structured-output", (
        "You are an assistant that follows output-format instructions exactly. "
        "Produce only the requested artefact, with no surrounding commentary."),
     "Tests whether a format-discipline directive improves format compliance."),
]


class SystemPromptOptimization(Optimization):
    id = "system_prompt"
    name = "System prompt"
    category = "system_prompt"
    description = (
        "Compares system-prompt strategies (minimal, accuracy-focused, concise, "
        "structured-output, plus a task-specific one) against sending no system prompt."
    )
    parameters = ["system"]
    fine_budget = 1

    def applicability(self, ctx: OptimizationContext) -> Applicability:
        if not ctx.capabilities.get("supports_system_prompt", True):
            return Applicability.skip(
                "This model's template does not appear to accept a system message, so "
                "system-prompt variants would be silently ignored."
            )
        if ctx.system_prompt:
            return Applicability.ok(
                "A system prompt was supplied; the alternatives below are compared "
                "against it."
            )
        return Applicability.ok("The model template accepts a system message.")

    def generate_configurations(self, ctx: OptimizationContext) -> List[RunConfig]:
        base = ctx.baseline_options
        configs: List[RunConfig] = []
        for key, label, text, rationale in SYSTEM_PROMPT_VARIANTS:
            configs.append(RunConfig(
                key=f"sys_{key}",
                label=f"System prompt: {label}",
                optimization_id=self.id,
                category=self.category,
                phase="coarse",
                options=dict(base),
                system_prompt=text,
                rationale=rationale,
                tags=["system_prompt", key],
            ))

        task_specific = _task_system_prompt(ctx)
        if task_specific:
            configs.append(RunConfig(
                key="sys_task",
                label="System prompt: task-specific",
                optimization_id=self.id,
                category=self.category,
                phase="coarse",
                options=dict(base),
                system_prompt=task_specific,
                rationale=(
                    "A system prompt written for this specific task type. Tests whether "
                    "task-specific framing beats the generic variants for this workload."
                ),
                tags=["system_prompt", "task"],
            ))
        return configs

    def refine(self, ctx: OptimizationContext,
               results: List[ConfigOutcome]) -> List[RunConfig]:
        best = best_by_quality(results)
        if best is None or not best.config.system_prompt:
            return []
        return [RunConfig(
            key=f"{best.config.key}_lowtemp",
            label=f"{best.config.label} + temperature 0.2",
            optimization_id=self.id,
            category=self.category,
            phase="fine",
            options={**ctx.baseline_options, "temperature": 0.2},
            system_prompt=best.config.system_prompt,
            rationale=(
                f"'{best.config.label}' led the system-prompt comparison "
                f"(quality {best.quality:.2f}); retested at a lower temperature to see "
                "whether the two effects combine."
            ),
            tags=["fine", "system_prompt"],
        )]


def _task_system_prompt(ctx: OptimizationContext) -> Optional[str]:
    exp = ctx.expectations or {}
    fmt = exp.get("format")
    reqs = exp.get("requirements") or []
    if fmt == "json":
        keys = exp.get("json_required_keys") or []
        key_part = (" with the keys " + ", ".join(keys)) if keys else ""
        return ("You are a data-extraction service. You return one valid JSON object"
                + key_part + " and nothing else - no prose, no markdown fence.")
    if fmt in ("bullets", "lines") and reqs:
        return ("You are an assistant that produces exactly the requested structure. "
                "Constraints for this task: " + "; ".join(reqs[:5]) + ".")
    if fmt == "code":
        return ("You are a software engineer. You return complete, runnable code in a "
                "single code block, with no explanation around it.")
    if reqs:
        return ("You are a precise assistant. For this task you must: "
                + "; ".join(reqs[:5]) + ".")
    return None
