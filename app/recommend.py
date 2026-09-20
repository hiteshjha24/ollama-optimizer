"""Recommendation engine.

A configuration is recommended by combining normalised scores with the weights
the user chose. Every step is exposed: the raw metric, the normalisation range,
the weight, and the contribution to the final number. The wording used
throughout the app and the report is deliberately conditional - the winner is
the configuration that scored highest *under these weights, on this prompt, on
this machine*, not a universally optimal setting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .metrics import percent_change, welch_t

OBJECTIVES: Dict[str, Dict[str, float]] = {
    "quality_first": {"quality": 0.60, "consistency": 0.25, "speed": 0.15},
    "speed_first": {"speed": 0.60, "quality": 0.20, "efficiency": 0.10, "consistency": 0.10},
    "balanced": {"quality": 0.40, "speed": 0.30, "consistency": 0.30},
}

OBJECTIVE_LABELS = {
    "quality_first": "Quality first",
    "speed_first": "Speed first",
    "balanced": "Balanced",
    "custom": "Custom weights",
}

METRIC_LABELS = {
    "quality": "Quality (0-10 heuristic score)",
    "speed": "Speed (tokens/sec and median latency)",
    "consistency": "Consistency across repeated runs",
    "efficiency": "Output produced per second of wall time",
}


def resolve_weights(objective: str, custom: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Return weights that sum to 1.0."""
    if objective == "custom" and custom:
        clean = {k: float(v) for k, v in custom.items()
                 if k in METRIC_LABELS and isinstance(v, (int, float)) and float(v) > 0}
        total = sum(clean.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in clean.items()}
    return dict(OBJECTIVES.get(objective, OBJECTIVES["balanced"]))


def _normalise(values: Dict[str, Optional[float]], higher_is_better: bool) -> Dict[str, Any]:
    """Min-max normalise to 0-10 across configurations."""
    valid = {k: v for k, v in values.items() if isinstance(v, (int, float))}
    if not valid:
        return {"scores": {}, "min": None, "max": None,
                "note": "metric unavailable for every configuration"}
    lo, hi = min(valid.values()), max(valid.values())
    scores: Dict[str, float] = {}
    for key, value in valid.items():
        if hi == lo:
            scores[key] = 10.0
        else:
            ratio = (value - lo) / (hi - lo)
            scores[key] = round(10.0 * (ratio if higher_is_better else 1 - ratio), 3)
    return {
        "scores": scores,
        "min": round(lo, 4),
        "max": round(hi, 4),
        "note": ("all configurations produced the same value, so each scores 10"
                 if hi == lo else None),
    }


def build_scoreboard(configs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute normalised component scores for every completed configuration.

    ``configs`` are configuration rows with their ``aggregate`` block attached.
    """
    usable = [c for c in configs
              if c.get("aggregate", {}).get("successful_runs", 0) > 0]

    quality_raw = {c["id"]: (c["aggregate"].get("quality") or {}).get("mean") for c in usable}
    tps_raw = {c["id"]: (c["aggregate"].get("tokens_per_second") or {}).get("mean") for c in usable}
    latency_raw = {c["id"]: (c["aggregate"].get("latency") or {}).get("median") for c in usable}
    consistency_raw = {c["id"]: (c["aggregate"].get("consistency") or {}).get("score") for c in usable}
    efficiency_raw = {}
    for c in usable:
        tokens = (c["aggregate"].get("output_tokens") or {}).get("mean")
        latency = (c["aggregate"].get("latency") or {}).get("mean")
        efficiency_raw[c["id"]] = (tokens / latency) if (tokens and latency) else None

    norm_quality = _normalise(quality_raw, True)
    norm_tps = _normalise(tps_raw, True)
    norm_latency = _normalise(latency_raw, False)
    norm_consistency = _normalise(consistency_raw, True)
    norm_efficiency = _normalise(efficiency_raw, True)

    speed_scores: Dict[str, float] = {}
    for cfg in usable:
        cid = cfg["id"]
        parts = []
        if cid in norm_tps["scores"]:
            parts.append((norm_tps["scores"][cid], 0.6))
        if cid in norm_latency["scores"]:
            parts.append((norm_latency["scores"][cid], 0.4))
        if parts:
            total_w = sum(w for _, w in parts)
            speed_scores[cid] = round(sum(s * w for s, w in parts) / total_w, 3)

    components = {
        "quality": {"scores": norm_quality["scores"], "raw": quality_raw,
                    "range": [norm_quality["min"], norm_quality["max"]],
                    "direction": "higher is better",
                    "source": "mean heuristic quality score across runs"},
        "speed": {"scores": speed_scores, "raw": {k: tps_raw.get(k) for k in speed_scores},
                  "range": [norm_tps["min"], norm_tps["max"]],
                  "direction": "higher is better",
                  "source": "60% normalised tokens/sec + 40% normalised (inverted) median latency"},
        "consistency": {"scores": norm_consistency["scores"], "raw": consistency_raw,
                        "range": [norm_consistency["min"], norm_consistency["max"]],
                        "direction": "higher is better",
                        "source": "pairwise output similarity and quality-score stability"},
        "efficiency": {"scores": norm_efficiency["scores"], "raw": efficiency_raw,
                       "range": [norm_efficiency["min"], norm_efficiency["max"]],
                       "direction": "higher is better",
                       "source": "mean output tokens divided by mean wall-clock latency"},
    }
    return {"components": components,
            "config_ids": [c["id"] for c in usable],
            "normalisation": "min-max across the configurations in this experiment, scaled 0-10"}


def score_configurations(configs: Sequence[Dict[str, Any]], weights: Dict[str, float]
                         ) -> List[Dict[str, Any]]:
    """Apply the weights and return configurations ranked by composite score."""
    board = build_scoreboard(configs)
    components = board["components"]
    by_id = {c["id"]: c for c in configs}

    rows: List[Dict[str, Any]] = []
    for cid in board["config_ids"]:
        cfg = by_id[cid]
        aggregate = cfg.get("aggregate", {})
        breakdown = []
        total = 0.0
        used_weight = 0.0
        for metric, weight in weights.items():
            score = components.get(metric, {}).get("scores", {}).get(cid)
            if score is None:
                breakdown.append({"metric": metric, "weight": weight, "score": None,
                                  "contribution": None,
                                  "note": "metric unavailable; weight redistributed"})
                continue
            contribution = score * weight
            total += contribution
            used_weight += weight
            breakdown.append({"metric": metric, "weight": round(weight, 4),
                              "score": round(score, 3),
                              "raw": components[metric]["raw"].get(cid),
                              "contribution": round(contribution, 4), "note": None})

        composite = (total / used_weight) if used_weight else None
        failure_rate = float(aggregate.get("failure_rate") or 0.0)
        penalty = 1.0 - min(1.0, failure_rate)
        penalised = round(composite * penalty, 4) if composite is not None else None

        rows.append({
            "configuration_id": cid,
            "label": cfg.get("label"),
            "category": cfg.get("category"),
            "optimization_id": cfg.get("optimization_id"),
            "phase": cfg.get("phase"),
            "is_baseline": bool(cfg.get("is_baseline")),
            "composite_score": penalised,
            "composite_before_penalty": round(composite, 4) if composite is not None else None,
            "failure_penalty": round(penalty, 3),
            "failure_rate": failure_rate,
            "used_weight": round(used_weight, 4),
            "breakdown": breakdown,
            "aggregate": aggregate,
            "options": cfg.get("options", {}),
            "prompt_modified": bool(cfg.get("prompt_modified")),
            "system_prompt": cfg.get("system_prompt"),
        })

    rows.sort(key=lambda r: (r["composite_score"] is not None, r["composite_score"] or 0),
              reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def compare_to_baseline(row: Dict[str, Any], baseline: Optional[Dict[str, Any]]
                        ) -> Dict[str, Any]:
    """Percentage deltas plus an honest significance note."""
    if not baseline:
        return {"available": False,
                "reason": "the baseline configuration produced no successful runs"}

    agg, base_agg = row.get("aggregate", {}), baseline.get("aggregate", {})

    def pick(block: Dict[str, Any], key: str, stat: str) -> Optional[float]:
        return (block.get(key) or {}).get(stat)

    quality_delta = percent_change(pick(agg, "quality", "mean"), pick(base_agg, "quality", "mean"))
    tps_delta = percent_change(pick(agg, "tokens_per_second", "mean"),
                               pick(base_agg, "tokens_per_second", "mean"))
    latency_delta = percent_change(pick(agg, "latency", "median"),
                                   pick(base_agg, "latency", "median"))
    consistency_delta = percent_change((agg.get("consistency") or {}).get("score"),
                                       (base_agg.get("consistency") or {}).get("score"))

    significance = welch_t(agg.get("samples", {}).get("quality", []),
                           base_agg.get("samples", {}).get("quality", []))
    latency_sig = welch_t(agg.get("samples", {}).get("latency", []),
                          base_agg.get("samples", {}).get("latency", []))

    return {
        "available": True,
        "baseline_configuration_id": baseline.get("configuration_id") or baseline.get("id"),
        "quality_percent": quality_delta,
        "tokens_per_second_percent": tps_delta,
        "median_latency_percent": latency_delta,
        "consistency_percent": consistency_delta,
        "quality_significance": significance,
        "latency_significance": latency_sig,
        "interpretation": _interpret(quality_delta, latency_delta, significance),
    }


def _interpret(quality_delta: Optional[float], latency_delta: Optional[float],
               significance: Dict[str, Any]) -> str:
    if quality_delta is None:
        return "Not enough data to compare quality with the baseline."
    direction = "higher" if quality_delta > 0 else "lower" if quality_delta < 0 else "unchanged"
    parts = [f"Quality is {abs(quality_delta):.1f}% {direction} than baseline"]
    if latency_delta is not None:
        faster = "faster" if latency_delta < 0 else "slower"
        parts.append(f"median latency is {abs(latency_delta):.1f}% {faster}")
    if significance.get("significant_95") is False:
        parts.append("the quality difference is within run-to-run noise at this sample size")
    elif significance.get("significant_95") is True:
        parts.append("the quality difference exceeds run-to-run noise (Welch's t, indicative only)")
    return "; ".join(parts) + "."


def recommend(configs: Sequence[Dict[str, Any]], objective: str,
              custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Full recommendation payload: ranking, winner, alternatives, and the maths."""
    weights = resolve_weights(objective, custom_weights)
    ranked = score_configurations(configs, weights)

    if not ranked:
        return {
            "objective": objective,
            "objective_label": OBJECTIVE_LABELS.get(objective, objective),
            "weights": weights,
            "ranking": [],
            "recommended": None,
            "alternatives": {},
            "reason": "No configuration produced a successful run, so nothing can be recommended.",
        }

    baseline_row = next((r for r in ranked if r["is_baseline"]), None)
    winner = ranked[0]

    def best_by(metric: str) -> Optional[Dict[str, Any]]:
        board = build_scoreboard(configs)["components"].get(metric, {}).get("scores", {})
        if not board:
            return None
        best_id = max(board, key=lambda k: board[k])
        return next((r for r in ranked if r["configuration_id"] == best_id), None)

    alternatives: Dict[str, Any] = {}
    for name, metric in (("maximum_quality", "quality"), ("maximum_speed", "speed"),
                         ("most_consistent", "consistency")):
        candidate = best_by(metric)
        if candidate and candidate["configuration_id"] != winner["configuration_id"]:
            alternatives[name] = {
                "configuration_id": candidate["configuration_id"],
                "label": candidate["label"],
                "composite_score": candidate["composite_score"],
                "comparison": compare_to_baseline(candidate, baseline_row),
                "basis": f"highest normalised {METRIC_LABELS.get(metric, metric)} in this experiment",
            }

    winner_comparison = compare_to_baseline(winner, baseline_row)
    tradeoffs = _tradeoffs(ranked, baseline_row)

    return {
        "objective": objective,
        "objective_label": OBJECTIVE_LABELS.get(objective, objective),
        "weights": weights,
        "weight_labels": {k: METRIC_LABELS.get(k, k) for k in weights},
        "normalisation": "min-max across this experiment's configurations, scaled 0-10",
        "ranking": ranked,
        "recommended": winner,
        "recommended_comparison": winner_comparison,
        "baseline_row": baseline_row,
        "alternatives": alternatives,
        "tradeoffs": tradeoffs,
        "explanation": _explain(winner, weights, baseline_row, winner_comparison),
        "caveat": (
            "Based on the selected evaluation weights and this benchmark's results, this "
            "configuration achieved the highest composite score. It is not a universally "
            "optimal setting: a different prompt, a different objective or different "
            "hardware can change the ranking."
        ),
    }


def _explain(winner: Dict[str, Any], weights: Dict[str, float],
             baseline: Optional[Dict[str, Any]], comparison: Dict[str, Any]) -> str:
    parts: List[str] = []
    for item in winner["breakdown"]:
        if item["score"] is None:
            continue
        parts.append(
            f"{METRIC_LABELS.get(item['metric'], item['metric'])} scored "
            f"{item['score']:.2f}/10 and carries weight {item['weight']:.0%}, "
            f"contributing {item['contribution']:.2f}"
        )
    math_line = "; ".join(parts)
    total = (f" The weighted contributions sum to {winner['composite_before_penalty']:.2f} over "
             f"{winner['used_weight']:.0%} of the available weight, giving a composite score of "
             f"{winner['composite_score']:.2f}/10.") if winner["composite_score"] is not None else ""
    if winner["is_baseline"]:
        lead = ("The baseline configuration ranked first: no tested optimization beat the "
                "model's own defaults under these weights. ")
    else:
        lead = f"'{winner['label']}' ranked first. "
    tail = ""
    if comparison.get("available"):
        tail = " Against baseline: " + comparison["interpretation"]
    return lead + math_line + "." + total + tail


def _tradeoffs(ranked: Sequence[Dict[str, Any]], baseline: Optional[Dict[str, Any]],
               limit: int = 6) -> List[Dict[str, Any]]:
    """Plus/minus summary of the leading configurations against baseline."""
    if not baseline:
        return []
    out = []
    for row in ranked[:limit]:
        if row["is_baseline"]:
            continue
        comparison = compare_to_baseline(row, baseline)
        pluses, minuses = [], []
        mapping = [
            ("quality_percent", "quality", True),
            ("tokens_per_second_percent", "tokens/sec", True),
            ("median_latency_percent", "median latency", False),
            ("consistency_percent", "consistency", True),
        ]
        for key, label, higher_better in mapping:
            value = comparison.get(key)
            if value is None or abs(value) < 1.0:
                continue
            improved = value > 0 if higher_better else value < 0
            text = f"{label} {value:+.1f}%"
            (pluses if improved else minuses).append(text)
        if pluses or minuses:
            out.append({
                "configuration_id": row["configuration_id"],
                "label": row["label"],
                "composite_score": row["composite_score"],
                "gains": pluses,
                "costs": minuses,
                "significance_note": comparison.get("quality_significance", {}).get("note"),
            })
    return out
