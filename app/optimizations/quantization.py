"""Quantization analysis.

This application never converts, re-quantizes, replaces or deletes a model.
Quantization is therefore handled in one of two ways:

* If **other locally installed models from the same family** carry a different
  quantization level, they are offered as a side-by-side comparison. That is a
  cross-model comparison, not an optimization of the selected model, and the
  report says so.
* Otherwise the section is informational: it reports the detected quantization
  and explains the trade-offs, with status "Not directly tested".
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import Applicability, Optimization, OptimizationContext, RunConfig

QUANT_NOTES = {
    "Q2": "Smallest and fastest, with the largest quality loss; usually only viable for very large models.",
    "Q3": "Aggressive compression; noticeable quality loss on small models.",
    "Q4": "The common default. Good size/quality balance for most local use.",
    "Q5": "Larger and slower than Q4 with a modest quality gain.",
    "Q6": "Close to Q8 quality at somewhat lower memory use.",
    "Q8": "Near-full-precision quality; roughly twice the memory of Q4.",
    "F16": "Half precision, no quantization loss; highest memory use.",
    "F32": "Full precision; rarely used for local inference.",
}


def _family_root(name: str) -> str:
    """'llama3.1:8b-instruct-q4_K_M' -> 'llama3.1'."""
    return (name or "").split(":")[0].strip().lower()


def describe_quantization(level: Optional[str]) -> Dict[str, Any]:
    if not level:
        return {
            "level": None,
            "note": "Ollama did not report a quantization level for this model.",
            "effects": [],
        }
    upper = str(level).upper()
    key = next((k for k in QUANT_NOTES if upper.startswith(k)), None)
    return {
        "level": level,
        "note": QUANT_NOTES.get(key, "Unrecognised quantization tag; no general guidance available."),
        "effects": [
            "Memory: lower-precision weights reduce resident size roughly in proportion "
            "to the bit width, which is often what decides whether a model fits in VRAM.",
            "Speed: smaller weights move less data per token, so heavily quantized models "
            "are usually faster when memory bandwidth is the bottleneck - but a model that "
            "already fits on the GPU may show little difference.",
            "Quality: quantization is lossy. The loss is usually small at Q5/Q6/Q8 and "
            "grows at Q3 and below, and it hits small models harder than large ones.",
        ],
    }


class QuantizationOptimization(Optimization):
    id = "quantization"
    name = "Quantization"
    category = "quantization"
    description = (
        "Reports the selected model's quantization and, when another locally installed "
        "variant of the same family uses a different quantization, benchmarks it "
        "side by side. No model files are ever created, converted or modified."
    )
    parameters = ["model variant"]
    fine_budget = 0

    def find_variants(self, ctx: OptimizationContext) -> List[Dict[str, Any]]:
        root = _family_root(ctx.model)
        current_quant = (ctx.capabilities.get("quantization_level") or "").upper()
        variants = []
        for model in ctx.available_models:
            name = model.get("name") or ""
            if name == ctx.model:
                continue
            if _family_root(name) != root:
                continue
            quant = (model.get("quantization_level") or "").upper()
            if not quant or quant == current_quant:
                continue
            variants.append(model)
        return variants[:2]

    def applicability(self, ctx: OptimizationContext) -> Applicability:
        if self.find_variants(ctx):
            return Applicability.ok(
                "Another locally installed model from the same family uses a different "
                "quantization level, so a direct comparison is possible."
            )
        return Applicability.info(
            "No alternative local quantized variant of this model family is installed, "
            "so quantization is reported as information only. This application does not "
            "download, convert or replace model files."
        )

    def generate_configurations(self, ctx: OptimizationContext) -> List[RunConfig]:
        configs: List[RunConfig] = []
        for variant in self.find_variants(ctx):
            name = variant["name"]
            quant = variant.get("quantization_level") or "unknown"
            size = variant.get("size_human") or "unknown size"
            configs.append(RunConfig(
                key=f"quant_{re.sub(r'[^a-zA-Z0-9]+', '_', name)}",
                label=f"Variant: {name} ({quant})",
                optimization_id=self.id,
                category=self.category,
                phase="coarse",
                options=dict(ctx.baseline_options),
                model_override=name,
                rationale=(
                    f"'{name}' is already installed locally, comes from the same family "
                    f"as the selected model, and is quantized at {quant} ({size}). It is "
                    "run on the identical prompt and settings so the quantization "
                    "difference can be seen directly. This is a comparison between two "
                    "separate models, not a change to the selected one."
                ),
                tags=["quantization", "cross_model"],
            ))
        return configs
