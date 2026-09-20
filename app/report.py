"""Report builder.

``build_report`` assembles one structured dictionary containing every section of
the report. The Markdown writer and the PDF generator both render that same
dictionary, so the two outputs can never disagree.

Every number in the report comes from the database rows produced by the
benchmark. Where a measurement is missing, the section says so explicitly.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import db
from .config import get_settings
from .evaluation import CRITERION_WEIGHTS
from .ollama import human_size
from .optimizations import STRATEGY_LABELS, describe_quantization, get_optimization
from .recommend import OBJECTIVE_LABELS, METRIC_LABELS, recommend
from .version import APP_NAME, APP_VERSION

MAX_OUTPUT_CHARS = 1800
RAW_OUTPUT_CONFIGS = 6


def _fmt(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def _stat(block: Optional[Dict[str, Any]], key: str = "mean", digits: int = 2,
          suffix: str = "") -> str:
    if not block or block.get("n", 0) == 0:
        return "N/A"
    return _fmt(block.get(key), digits, suffix)


def _timestamp(value: Optional[float]) -> str:
    if not value:
        return "N/A"
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_report(experiment_id: str) -> Dict[str, Any]:
    """Assemble the complete report payload for one experiment."""
    experiment = db.get_experiment(experiment_id)
    if not experiment:
        raise ValueError(f"Experiment {experiment_id} not found.")

    configs = db.list_configurations(experiment_id)
    runs_by_config: Dict[str, List[Dict[str, Any]]] = {}
    for cfg in configs:
        runs_by_config[cfg["id"]] = db.list_runs(experiment_id, cfg["id"])

    result = recommend(configs, experiment["objective"], experiment.get("weights"))
    baseline_cfg = next((c for c in configs if c["is_baseline"]), None)
    summary = experiment.get("summary") or {}

    report: Dict[str, Any] = {
        "meta": {
            "experiment_id": experiment_id,
            "title": f"Optimization benchmark: {experiment['model']}",
            "generated_at": time.time(),
            "generated_at_text": _timestamp(time.time()),
            "app": APP_NAME,
            "app_version": APP_VERSION,
            "ollama_version": experiment.get("ollama_version") or "unknown",
        },
        "experiment": experiment,
        "configs": configs,
        "runs_by_config": runs_by_config,
        "recommendation": result,
    }

    report["executive_summary"] = _executive_summary(experiment, configs, result, summary)
    report["model_info"] = _model_info(experiment)
    report["methodology"] = _methodology(experiment, configs)
    report["baseline"] = _baseline_section(baseline_cfg, runs_by_config)
    report["optimization_results"] = _optimization_sections(experiment, configs, result,
                                                            baseline_cfg)
    report["comparison"] = _comparison_table(result)
    report["tradeoffs"] = result.get("tradeoffs", [])
    report["recommended"] = _recommended_section(result, experiment)
    report["alternatives"] = result.get("alternatives", {})
    report["not_tested"] = summary.get("not_tested", [])
    report["raw_outputs"] = _raw_outputs(configs, runs_by_config, result)
    report["limitations"] = _limitations(experiment, configs)
    report["meta"]["estimated_pages"] = 0  # placeholder for the renderer
    report["meta"]["estimated_pages"] = _estimate_pages(report)
    return report


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def _executive_summary(experiment: Dict[str, Any], configs: List[Dict[str, Any]],
                       result: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    baseline = next((c for c in configs if c["is_baseline"]), None)
    base_agg = (baseline or {}).get("aggregate", {})
    winner = result.get("recommended")
    comparison = result.get("recommended_comparison", {})

    findings: List[str] = []
    if winner and winner["is_baseline"]:
        findings.append(
            "No tested optimization outscored the model's default configuration under "
            "the selected weights. The defaults are a reasonable choice for this prompt."
        )
    elif winner:
        findings.append(
            f"'{winner['label']}' achieved the highest composite score "
            f"({_fmt(winner['composite_score'])}/10) under the "
            f"{OBJECTIVE_LABELS.get(experiment['objective'], experiment['objective'])} weighting."
        )
        if comparison.get("available"):
            findings.append(comparison["interpretation"])

    categories = sorted({c["category"] for c in configs if not c["is_baseline"]})
    failed = [c for c in configs if c["aggregate"].get("failed_runs")]
    if failed:
        findings.append(
            f"{len(failed)} configuration(s) had at least one failed run; those runs are "
            "excluded from the statistics and listed in the raw-output section."
        )
    significance = (comparison.get("quality_significance") or {}).get("significant_95")
    if significance is False:
        findings.append(
            "The quality difference between the recommended configuration and the "
            "baseline is within run-to-run variation at this sample size. Treat it as "
            "a weak signal and increase runs per configuration to confirm."
        )

    return {
        "model": experiment["model"],
        "benchmark_date": _timestamp(experiment.get("started_at") or experiment["created_at"]),
        "duration_seconds": (round((experiment.get("finished_at") or time.time())
                                   - (experiment.get("started_at") or experiment["created_at"]), 1)),
        "prompt_label": experiment.get("prompt_label") or "Custom prompt",
        "prompt_category": experiment.get("prompt_category") or "custom",
        "strategy": STRATEGY_LABELS.get(experiment["strategy"], experiment["strategy"]),
        "objective": OBJECTIVE_LABELS.get(experiment["objective"], experiment["objective"]),
        "configurations_tested": len(configs),
        "total_runs": sum(c["aggregate"].get("total_runs", 0) for c in configs),
        "categories_tested": categories,
        "baseline": {
            "quality": _stat(base_agg.get("quality")),
            "median_latency": _stat(base_agg.get("latency"), "median"),
            "tokens_per_second": _stat(base_agg.get("tokens_per_second"), "mean", 1),
            "consistency": _fmt((base_agg.get("consistency") or {}).get("score")),
        },
        "recommended": winner,
        "comparison": comparison,
        "major_findings": findings,
        "caveat": result.get("caveat"),
    }


def _model_info(experiment: Dict[str, Any]) -> Dict[str, Any]:
    meta = experiment.get("model_meta") or {}
    host = experiment.get("host_info") or {}
    quant = describe_quantization(meta.get("quantization_level"))
    return {
        "name": experiment["model"],
        "family": meta.get("family") or "not reported",
        "families": meta.get("families") or [],
        "parameter_size": meta.get("parameter_size") or "not reported",
        "quantization_level": meta.get("quantization_level") or "not reported",
        "quantization_note": quant["note"],
        "quantization_effects": quant["effects"],
        "format": meta.get("format") or "not reported",
        "size": meta.get("size_human") or human_size(meta.get("size_bytes")) or "not reported",
        "digest": meta.get("digest") or "not reported",
        "modified_at": meta.get("modified_at") or "not reported",
        "ollama_version": experiment.get("ollama_version") or "unknown",
        "host": {
            "platform": host.get("platform") or "not reported",
            "cpu": (f"{host.get('cpu_count_physical') or '?'} physical / "
                    f"{host.get('cpu_count_logical') or '?'} logical cores"),
            "processor": host.get("processor") or "not reported",
            "ram_total_gb": host.get("ram_total_gb"),
            "gpu": host.get("gpu"),
            "python": host.get("python"),
        },
    }


def _methodology(experiment: Dict[str, Any], configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    phases: Dict[str, int] = {}
    for cfg in configs:
        phases[cfg["phase"]] = phases.get(cfg["phase"], 0) + 1
    return {
        "prompt": experiment["prompt"],
        "prompt_label": experiment.get("prompt_label") or "Custom prompt",
        "system_prompt": experiment.get("system_prompt"),
        "runs_per_config": experiment["runs_per_config"],
        "max_tokens": experiment["max_tokens"],
        "timeout_seconds": experiment["timeout_seconds"],
        "streaming": experiment["stream"],
        "concurrency": experiment["concurrency"],
        "seed": experiment.get("seed"),
        "evaluator_mode": experiment["evaluator_mode"],
        "configurations": len(configs),
        "phase_counts": phases,
        "search_strategy": [
            "Baseline: the model's own defaults with only an output-token cap, measured first.",
            "Warm-up: one unmeasured generation loads the model so the first measured run "
            "does not include load time.",
            "Coarse search: each generation parameter is varied one at a time (OFAT), so "
            "cost grows linearly with the number of parameters instead of multiplicatively.",
            "Fine search: the best value of each parameter is combined, and the most "
            "influential parameter is probed at neighbouring values.",
            "Validation: the top-ranked configurations are re-sampled with additional runs "
            "to strengthen their statistics.",
        ],
        "metrics_collected": [
            "Wall-clock latency measured by the client",
            "Time to first token (streaming runs only)",
            "Ollama total_duration, load_duration, prompt_eval_duration, eval_duration",
            "Prompt tokens and generated tokens as reported by Ollama",
            "Tokens per second (generated tokens / eval duration)",
            "System CPU and RAM during each run (psutil, system-wide)",
            "GPU utilisation and memory (nvidia-smi, when present)",
        ],
        "evaluation_method": {
            "primary": "Deterministic heuristics over the produced text.",
            "criteria": {name: f"weight {weight:.0%}" for name, weight in CRITERION_WEIGHTS.items()},
            "notes": [
                "Correctness is only scored where the prompt has a verifiable answer; for "
                "open-ended prompts it is reported as not scorable rather than estimated.",
                "Instruction following counts explicit, checkable constraints (word limits, "
                "line counts, forbidden words, required format).",
                "Consistency combines pairwise content-word similarity across runs with the "
                "stability of the quality score.",
                "Scores measure observable output only. They do not assess hidden reasoning, "
                "and for open-ended prompts they do not verify factual accuracy.",
            ],
            "llm_judge": (
                "An optional second evaluator uses another Ollama model. Its scores are "
                "labelled as model-generated estimates and never replace the deterministic "
                "metrics."
                if experiment["evaluator_mode"] == "heuristic+llm"
                else "Not used in this experiment."
            ),
        },
        "limitations_inline": (
            f"{experiment['runs_per_config']} runs per configuration is enough to expose "
            "gross differences but not small ones; confidence intervals are reported "
            "alongside every mean."
        ),
    }


def _baseline_section(baseline: Optional[Dict[str, Any]],
                      runs_by_config: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not baseline:
        return {"available": False,
                "reason": "No baseline configuration was recorded for this experiment."}
    agg = baseline["aggregate"]
    runs = runs_by_config.get(baseline["id"], [])
    resources = agg.get("resources") or {}
    return {
        "available": True,
        "configuration_id": baseline["id"],
        "label": baseline["label"],
        "options": baseline["options"],
        "rationale": baseline.get("rationale"),
        "runs": agg.get("total_runs", 0),
        "successful_runs": agg.get("successful_runs", 0),
        "failed_runs": agg.get("failed_runs", 0),
        "metrics": [
            ("Quality (0-10)", _stat(agg.get("quality")), _ci(agg.get("quality"))),
            ("Median latency", _stat(agg.get("latency"), "median", 3, " s"), ""),
            ("Mean latency", _stat(agg.get("latency"), "mean", 3, " s"), _ci(agg.get("latency"), " s")),
            ("Latency min / max", _range(agg.get("latency"), " s"), ""),
            ("Time to first token", _stat(agg.get("time_to_first_token"), "mean", 3, " s"), ""),
            ("Tokens per second", _stat(agg.get("tokens_per_second"), "mean", 2), _ci(agg.get("tokens_per_second"))),
            ("Generated tokens", _stat(agg.get("output_tokens"), "mean", 1), ""),
            ("Prompt tokens", _stat(agg.get("prompt_tokens"), "mean", 1), ""),
            ("Total duration (Ollama)", _stat(agg.get("total_duration"), "mean", 3, " s"), ""),
            ("Prompt eval duration", _stat(agg.get("prompt_eval_duration"), "mean", 3, " s"), ""),
            ("Generation duration", _stat(agg.get("eval_duration"), "mean", 3, " s"), ""),
            ("Load duration", _stat(agg.get("load_duration"), "mean", 3, " s"), ""),
            ("Consistency (0-10)", _fmt((agg.get("consistency") or {}).get("score")), ""),
            ("Failure rate", f"{agg.get('failure_rate', 0) * 100:.0f}%", ""),
            ("CPU during runs", _resource(resources, "cpu_percent_mean", "%"), ""),
            ("RAM during runs", _resource(resources, "ram_percent_mean", "%"), ""),
            ("GPU utilisation", _resource(resources, "gpu_utilization_mean", "%"), ""),
            ("GPU memory used", _resource(resources, "gpu_memory_used_mb_mean", " MB"), ""),
        ],
        "criteria": agg.get("criteria", {}),
        "sample_output": _first_output(runs),
        "errors": agg.get("errors", []),
    }


def _optimization_sections(experiment: Dict[str, Any], configs: List[Dict[str, Any]],
                           result: Dict[str, Any],
                           baseline: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from .recommend import compare_to_baseline
    ranking_by_id = {r["configuration_id"]: r for r in result.get("ranking", [])}
    baseline_row = result.get("baseline_row")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for cfg in configs:
        if cfg["is_baseline"]:
            continue
        grouped.setdefault(cfg["optimization_id"], []).append(cfg)

    sections = []
    for opt_id, items in grouped.items():
        opt = get_optimization(opt_id)
        rows = []
        for cfg in items:
            agg = cfg["aggregate"]
            row_score = ranking_by_id.get(cfg["id"])
            comparison = (compare_to_baseline(row_score, baseline_row)
                          if row_score and baseline_row else {"available": False})
            rows.append({
                "configuration_id": cfg["id"],
                "label": cfg["label"],
                "phase": cfg["phase"],
                "options": {k: v for k, v in cfg["options"].items() if not k.startswith("_")},
                "switches": {k.lstrip("_"): v for k, v in cfg["options"].items()
                             if k.startswith("_")},
                "rationale": cfg.get("rationale"),
                "prompt_changed": bool(cfg.get("prompt")
                                       and cfg["prompt"] != experiment["prompt"]),
                "prompt": cfg.get("prompt"),
                "system_prompt": cfg.get("system_prompt"),
                "quality": _stat(agg.get("quality")),
                "quality_ci": _ci(agg.get("quality")),
                "latency_median": _stat(agg.get("latency"), "median", 3, " s"),
                "tokens_per_second": _stat(agg.get("tokens_per_second"), "mean", 1),
                "consistency": _fmt((agg.get("consistency") or {}).get("score")),
                "failure_rate": f"{agg.get('failure_rate', 0) * 100:.0f}%",
                "composite": _fmt((row_score or {}).get("composite_score")),
                "comparison": comparison,
                "aggregate": agg,
            })
        rows.sort(key=lambda r: (r["composite"] == "N/A", r["composite"]), reverse=False)

        best = max((r for r in rows if r["quality"] != "N/A"),
                   key=lambda r: float(r["quality"]), default=None)
        sections.append({
            "optimization_id": opt_id,
            "name": opt.name if opt else opt_id,
            "category": items[0]["category"],
            "description": opt.description if opt else "",
            "what_changed": _what_changed(opt_id, items),
            "why_tested": opt.description if opt else "",
            "configurations": rows,
            "best": best,
            "summary": _category_summary(opt_id, rows, best),
        })
    sections.sort(key=lambda s: s["name"])
    return sections


def _what_changed(opt_id: str, items: List[Dict[str, Any]]) -> str:
    if opt_id == "generation_params":
        params = sorted({k for c in items for k in c["options"] if k != "num_predict"})
        return "Sampling parameters varied: " + ", ".join(params) + "."
    if opt_id == "prompt_optimization":
        return ("The prompt text was rewritten; the model, sampling settings and system "
                "prompt were held constant.")
    if opt_id == "system_prompt":
        return ("A system message was supplied; the user prompt and sampling settings "
                "were held constant.")
    if opt_id == "context":
        return ("The context window (num_ctx) and/or the layout of the prompt content "
                "were changed.")
    if opt_id == "runtime":
        return ("Runtime settings were changed: streaming mode, CPU thread count and "
                "batch size.")
    if opt_id == "quantization":
        return ("A different locally installed model of the same family was run on the "
                "identical prompt. This compares two models, not two settings of one.")
    return "See the per-configuration rationale."


def _category_summary(opt_id: str, rows: List[Dict[str, Any]],
                      best: Optional[Dict[str, Any]]) -> str:
    if not rows:
        return "No configurations were run in this category."
    if not best:
        return "No configuration in this category produced a scorable result."
    comparison = best.get("comparison") or {}
    if comparison.get("available"):
        return (f"The strongest configuration in this category was '{best['label']}' "
                f"(quality {best['quality']}/10, median latency {best['latency_median']}). "
                + comparison.get("interpretation", ""))
    return (f"The strongest configuration in this category was '{best['label']}' "
            f"(quality {best['quality']}/10).")


def _comparison_table(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for row in result.get("ranking", []):
        agg = row.get("aggregate", {})
        component = {b["metric"]: b for b in row["breakdown"]}
        rows.append({
            "rank": row["rank"],
            "configuration_id": row["configuration_id"],
            "label": row["label"] + (" [baseline]" if row["is_baseline"] else ""),
            "category": row["category"],
            "quality": _stat(agg.get("quality")),
            "speed_score": _fmt((component.get("speed") or {}).get("score")),
            "consistency": _fmt((agg.get("consistency") or {}).get("score")),
            "avg_latency": _stat(agg.get("latency"), "mean", 2, "s"),
            "median_latency": _stat(agg.get("latency"), "median", 2, "s"),
            "tokens_per_second": _stat(agg.get("tokens_per_second"), "mean", 1),
            "runs": agg.get("successful_runs", 0),
            "failures": agg.get("failed_runs", 0),
            "composite": _fmt(row["composite_score"]),
            "is_baseline": row["is_baseline"],
        })
    return {
        "rows": rows,
        "note": ("Composite is the weighted combination of the normalised component "
                 "scores; the raw measurements behind it are in the columns to its left "
                 "and in the per-run detail."),
        "weights": result.get("weights", {}),
    }


def _recommended_section(result: Dict[str, Any], experiment: Dict[str, Any]) -> Dict[str, Any]:
    winner = result.get("recommended")
    if not winner:
        return {"available": False, "reason": result.get("reason", "No scorable configuration.")}
    agg = winner.get("aggregate", {})
    comparison = result.get("recommended_comparison", {})
    options = {k: v for k, v in (winner.get("options") or {}).items() if not k.startswith("_")}
    return {
        "available": True,
        "model": experiment["model"],
        "objective": OBJECTIVE_LABELS.get(experiment["objective"], experiment["objective"]),
        "weights": {METRIC_LABELS.get(k, k): f"{v:.0%}" for k, v in result["weights"].items()},
        "label": winner["label"],
        "configuration_id": winner["configuration_id"],
        "options": options,
        "system_prompt": winner.get("system_prompt"),
        "prompt_modified": winner.get("prompt_modified"),
        "observed": {
            "quality": _stat(agg.get("quality")),
            "quality_ci": _ci(agg.get("quality")),
            "tokens_per_second": _stat(agg.get("tokens_per_second"), "mean", 1),
            "median_latency": _stat(agg.get("latency"), "median", 2, " s"),
            "consistency": _fmt((agg.get("consistency") or {}).get("score")),
            "runs": agg.get("successful_runs", 0),
        },
        "vs_baseline": comparison,
        "why": result.get("explanation"),
        "caveat": result.get("caveat"),
        "reversibility": (
            "These are request-time settings only. Nothing in your Ollama installation "
            "was changed by this benchmark. To use them, pass the options with your "
            "API request or set them in your client; no Modelfile edit is required."
        ),
        "how_to_apply": _how_to_apply(experiment["model"], options, winner.get("system_prompt")),
    }


def _how_to_apply(model: str, options: Dict[str, Any], system: Optional[str]) -> Dict[str, str]:
    import json as _json
    payload = {"model": model, "prompt": "YOUR PROMPT HERE", "stream": False,
               "options": options}
    if system:
        payload["system"] = system
    curl = ("curl http://localhost:11434/api/generate -d '"
            + _json.dumps(payload, indent=2) + "'")
    cli_lines = [f"ollama run {model}"]
    for key, value in options.items():
        if key == "num_predict":
            continue
        cli_lines.append(f"/set parameter {key} {value}")
    return {
        "api": curl,
        "cli": "\n".join(cli_lines),
        "note": ("The /set parameter commands apply to the current interactive session "
                 "only. They do not modify the stored model."),
    }


def _raw_outputs(configs: List[Dict[str, Any]],
                 runs_by_config: Dict[str, List[Dict[str, Any]]],
                 result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Representative outputs: baseline, the winner, then the next best."""
    order: List[str] = []
    baseline = next((c["id"] for c in configs if c["is_baseline"]), None)
    if baseline:
        order.append(baseline)
    for row in result.get("ranking", []):
        if row["configuration_id"] not in order:
            order.append(row["configuration_id"])
        if len(order) >= RAW_OUTPUT_CONFIGS:
            break

    by_id = {c["id"]: c for c in configs}
    out = []
    for cfg_id in order:
        cfg = by_id.get(cfg_id)
        if not cfg:
            continue
        runs = runs_by_config.get(cfg_id, [])
        entries = []
        for run in runs[:3]:
            evaluation = run.get("evaluation") or {}
            entries.append({
                "run_index": run["run_index"] + 1,
                "status": run["status"],
                "error": run.get("error"),
                "output": _truncate(run.get("output") or ""),
                "output_truncated_for_report": len(run.get("output") or "") > MAX_OUTPUT_CHARS,
                "metrics": run.get("metrics", {}),
                "quality": evaluation.get("quality_score"),
                "criteria": evaluation.get("criteria", {}),
            })
        out.append({
            "configuration_id": cfg_id,
            "label": cfg["label"] + (" [baseline]" if cfg["is_baseline"] else ""),
            "options": {k: v for k, v in cfg["options"].items() if not k.startswith("_")},
            "prompt": cfg.get("prompt"),
            "system_prompt": cfg.get("system_prompt"),
            "runs": entries,
        })
    return out


def _limitations(experiment: Dict[str, Any], configs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    runs = experiment["runs_per_config"]
    items = [
        {
            "title": "Sample size",
            "text": (f"Each configuration was measured {runs} times (top configurations "
                     f"were re-sampled with extra runs). That is enough to expose large "
                     "differences, but differences of a few percent are likely to be "
                     "noise. Confidence intervals are reported with every mean; where "
                     "they overlap, treat the configurations as indistinguishable."),
        },
        {
            "title": "Prompt dependence",
            "text": ("Results apply to the prompt that was benchmarked. A configuration "
                     "that helps a JSON-extraction prompt may hurt creative writing. "
                     "Re-run the benchmark with a prompt representative of your workload "
                     "before adopting a setting broadly."),
        },
        {
            "title": "Stochastic generation",
            "text": ("Unless a seed is fixed, the model samples differently on every run. "
                     "Some of the variation between configurations is sampling noise "
                     "rather than a real effect of the setting."),
        },
        {
            "title": "Evaluator limitations",
            "text": ("Quality scores are deterministic heuristics over the visible output: "
                     "format checks, explicit-constraint checks, answer-pattern matching "
                     "and repetition detection. They do not verify factual accuracy for "
                     "open-ended prompts and they do not measure reasoning. A human "
                     "reading the raw outputs may reasonably disagree with them."),
        },
        {
            "title": "Hardware dependence",
            "text": ("Latency and throughput reflect this machine and its load at the time "
                     "of the run. Results will differ on other hardware, and background "
                     "activity on this machine can move the numbers."),
        },
        {
            "title": "Runtime and version differences",
            "text": (f"Measured against Ollama {experiment.get('ollama_version') or 'unknown'}. "
                     "Other versions may change default parameters, scheduling, and which "
                     "options are honoured."),
        },
    ]
    missing = []
    for cfg in configs:
        agg = cfg.get("aggregate", {})
        if (agg.get("time_to_first_token") or {}).get("n", 0) == 0:
            missing.append("time to first token (non-streaming runs)")
            break
    resources = (configs[0].get("aggregate", {}).get("resources") if configs else {}) or {}
    if not resources.get("gpu_name"):
        missing.append("GPU utilisation and GPU memory (nvidia-smi unavailable)")
    if resources.get("cpu_percent_mean") is None:
        missing.append("CPU and RAM usage (psutil unavailable)")
    if missing:
        items.append({
            "title": "Unavailable metrics",
            "text": ("The following were not measurable on this system and are reported "
                     "as N/A rather than estimated: " + "; ".join(sorted(set(missing))) + "."),
        })
    items.append({
        "title": "Scope of changes",
        "text": ("Every configuration in this report is a request-time setting. No model "
                 "file, Modelfile, Ollama setting or system configuration was modified, "
                 "and nothing needs to be undone."),
    })
    return items


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _ci(block: Optional[Dict[str, Any]], suffix: str = "") -> str:
    if not block or not block.get("ci95"):
        return ""
    low, high = block["ci95"]
    return f"95% CI [{low:.2f}, {high:.2f}]{suffix}"


def _range(block: Optional[Dict[str, Any]], suffix: str = "") -> str:
    if not block or block.get("n", 0) == 0:
        return "N/A"
    return f"{block['min']:.3f} / {block['max']:.3f}{suffix}"


def _resource(resources: Dict[str, Any], key: str, suffix: str) -> str:
    value = (resources or {}).get(key)
    if value is None:
        return "N/A - metric unavailable on this system"
    return f"{value:.1f}{suffix}"


def _first_output(runs: List[Dict[str, Any]]) -> str:
    for run in runs:
        if run.get("status") == "ok" and run.get("output"):
            return _truncate(run["output"])
    return "(no successful output recorded)"


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n... [truncated for the report; full text is in the app]"


def _estimate_pages(report: Dict[str, Any]) -> int:
    """Estimate the page count from the rendered report text.

    Measured against the generated PDFs, roughly 2,600 rendered characters fill
    one A4 page at this template's font size, plus a cover page and the chart
    page. This is an estimate for the UI; the PDF writer reports the real count.
    """
    try:
        rendered = render_markdown(report)
    except Exception:  # never let the estimate break report building
        return 5
    return max(4, round(len(rendered) / 2600) + 2)


# --------------------------------------------------------------------------
# markdown rendering
# --------------------------------------------------------------------------

def render_markdown(report: Dict[str, Any]) -> str:
    exp = report["experiment"]
    summary = report["executive_summary"]
    lines: List[str] = []
    add = lines.append

    add(f"# {report['meta']['title']}")
    add("")
    add(f"*Generated by {report['meta']['app']} {report['meta']['app_version']} on "
        f"{report['meta']['generated_at_text']}*")
    add("")
    add(f"Experiment ID: `{report['meta']['experiment_id']}`")
    add("")

    # 1 executive summary
    add("## 1. Executive summary")
    add("")
    add(f"- **Model:** {summary['model']}")
    add(f"- **Benchmark date:** {summary['benchmark_date']}")
    add(f"- **Prompt:** {summary['prompt_label']} ({summary['prompt_category']})")
    add(f"- **Optimization scope:** {summary['strategy']}")
    add(f"- **Objective:** {summary['objective']}")
    add(f"- **Configurations tested:** {summary['configurations_tested']} "
        f"across {len(summary['categories_tested'])} categories")
    add(f"- **Total generations:** {summary['total_runs']}")
    add(f"- **Duration:** {summary['duration_seconds']} s")
    add("")
    add("**Baseline performance:** quality "
        f"{summary['baseline']['quality']}/10, median latency "
        f"{summary['baseline']['median_latency']} s, "
        f"{summary['baseline']['tokens_per_second']} tokens/s, consistency "
        f"{summary['baseline']['consistency']}/10.")
    add("")
    add("**Major findings:**")
    add("")
    for finding in summary["major_findings"]:
        add(f"- {finding}")
    add("")
    if summary.get("caveat"):
        add(f"> {summary['caveat']}")
        add("")

    # 2 model information
    info = report["model_info"]
    add("## 2. Model information")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    for label, key in [("Model", "name"), ("Family", "family"),
                       ("Parameters", "parameter_size"), ("Quantization", "quantization_level"),
                       ("Format", "format"), ("Size", "size"), ("Digest", "digest"),
                       ("Modified", "modified_at"), ("Ollama version", "ollama_version")]:
        add(f"| {label} | {info[key]} |")
    host = info["host"]
    add(f"| Host platform | {host['platform']} |")
    add(f"| CPU | {host['cpu']} |")
    add(f"| RAM | {host['ram_total_gb'] or 'not reported'} GB |")
    add(f"| GPU | {_gpu_text(host['gpu'])} |")
    add("")
    add(f"*Quantization note:* {info['quantization_note']}")
    add("")
    for effect in info["quantization_effects"]:
        add(f"- {effect}")
    add("")

    # 3 methodology
    method = report["methodology"]
    add("## 3. Methodology")
    add("")
    add(f"- Runs per configuration: **{method['runs_per_config']}**")
    add(f"- Maximum generated tokens: {method['max_tokens']}")
    add(f"- Timeout: {method['timeout_seconds']} s")
    add(f"- Streaming: {'on' if method['streaming'] else 'off'}")
    add(f"- Concurrency: {method['concurrency']} request(s) at a time")
    add(f"- Seed used for determinism tests: {method['seed']}")
    add(f"- Evaluator: {method['evaluator_mode']}")
    add("")
    add("**Search strategy**")
    add("")
    for index, step in enumerate(method["search_strategy"], start=1):
        add(f"{index}. {step}")
    add("")
    add("**Metrics collected**")
    add("")
    for metric in method["metrics_collected"]:
        add(f"- {metric}")
    add("")
    add("**Evaluation method**")
    add("")
    add(method["evaluation_method"]["primary"])
    add("")
    for name, weight in method["evaluation_method"]["criteria"].items():
        add(f"- {name}: {weight}")
    add("")
    for note in method["evaluation_method"]["notes"]:
        add(f"- {note}")
    add("")
    add(f"- LLM judge: {method['evaluation_method']['llm_judge']}")
    add("")
    add("**Test prompt**")
    add("")
    add("```text")
    add(method["prompt"][:4000])
    add("```")
    add("")

    # 4 baseline
    baseline = report["baseline"]
    add("## 4. Baseline results")
    add("")
    if not baseline.get("available"):
        add(baseline.get("reason", "No baseline available."))
        add("")
    else:
        add(f"Configuration: `{baseline['options']}` - {baseline['rationale']}")
        add("")
        add("| Measurement | Value | Uncertainty |")
        add("| --- | --- | --- |")
        for label, value, extra in baseline["metrics"]:
            add(f"| {label} | {value} | {extra or '-'} |")
        add("")
        add("**Representative baseline output**")
        add("")
        add("```text")
        add(baseline["sample_output"])
        add("```")
        add("")

    # 5 optimization results
    add("## 5. Optimization results")
    add("")
    for section in report["optimization_results"]:
        add(f"### 5.{report['optimization_results'].index(section) + 1} {section['name']}")
        add("")
        add(f"*What changed:* {section['what_changed']}")
        add("")
        add(f"*Why it was tested:* {section['why_tested']}")
        add("")
        add("| Configuration | Phase | Quality | Median latency | Tokens/s | Consistency | Failures |")
        add("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in section["configurations"]:
            add(f"| {row['label']} | {row['phase']} | {row['quality']} | "
                f"{row['latency_median']} | {row['tokens_per_second']} | "
                f"{row['consistency']} | {row['failure_rate']} |")
        add("")
        add(section["summary"])
        add("")
        for row in section["configurations"][:3]:
            if row.get("rationale"):
                add(f"- **{row['label']}** - {row['rationale']}")
        add("")

    # 6 comparison
    add("## 6. Comparative analysis")
    add("")
    add("| # | Configuration | Quality | Speed | Consistency | Avg latency | Median | Tokens/s | Composite |")
    add("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["comparison"]["rows"]:
        add(f"| {row['rank']} | {row['label']} | {row['quality']} | {row['speed_score']} | "
            f"{row['consistency']} | {row['avg_latency']} | {row['median_latency']} | "
            f"{row['tokens_per_second']} | {row['composite']} |")
    add("")
    add(report["comparison"]["note"])
    add("")

    # 7 tradeoffs
    add("## 7. Optimization trade-offs")
    add("")
    if not report["tradeoffs"]:
        add("No configuration differed from the baseline by more than 1% on any tracked "
            "metric, so there are no trade-offs to report.")
        add("")
    for item in report["tradeoffs"]:
        add(f"**{item['label']}**")
        add("")
        for gain in item["gains"]:
            add(f"- `+` {gain}")
        for cost in item["costs"]:
            add(f"- `-` {cost}")
        add("")

    # 8 recommendation
    rec = report["recommended"]
    add("## 8. Recommended configuration")
    add("")
    if not rec.get("available"):
        add(rec.get("reason", "No recommendation available."))
        add("")
    else:
        add(f"**Model:** {rec['model']}")
        add("")
        add(f"**Objective:** {rec['objective']} "
            f"({', '.join(f'{k} {v}' for k, v in rec['weights'].items())})")
        add("")
        add(f"**Configuration:** {rec['label']}")
        add("")
        add("```json")
        import json as _json
        add(_json.dumps(rec["options"], indent=2))
        add("```")
        if rec.get("system_prompt"):
            add("")
            add("System prompt used:")
            add("")
            add("```text")
            add(rec["system_prompt"])
            add("```")
        add("")
        add("**Observed results**")
        add("")
        observed = rec["observed"]
        add(f"- Quality: {observed['quality']}/10 {observed['quality_ci']}")
        add(f"- Tokens/sec: {observed['tokens_per_second']}")
        add(f"- Median latency: {observed['median_latency']}")
        add(f"- Consistency: {observed['consistency']}/10")
        add(f"- Successful runs: {observed['runs']}")
        add("")
        comparison = rec.get("vs_baseline") or {}
        if comparison.get("available"):
            add("**Compared with baseline**")
            add("")
            add(f"- Quality: {_pct(comparison.get('quality_percent'))}")
            add(f"- Tokens/sec: {_pct(comparison.get('tokens_per_second_percent'))}")
            add(f"- Median latency: {_pct(comparison.get('median_latency_percent'))}")
            add(f"- Consistency: {_pct(comparison.get('consistency_percent'))}")
            add("")
        add("**Why this configuration was selected**")
        add("")
        add(rec["why"])
        add("")
        add(f"> {rec['caveat']}")
        add("")
        add("**How to apply it**")
        add("")
        add("```bash")
        add(rec["how_to_apply"]["api"])
        add("```")
        add("")
        add(rec["reversibility"])
        add("")

    # 9 alternatives
    add("## 9. Alternative configurations")
    add("")
    if not report["alternatives"]:
        add("The recommended configuration also led on every individual metric, so there "
            "is no separate quality-first or speed-first alternative to offer.")
        add("")
    for name, alt in report["alternatives"].items():
        add(f"- **{name.replace('_', ' ').title()}:** {alt['label']} "
            f"(composite {_fmt(alt['composite_score'])}/10) - {alt['basis']}.")
    add("")

    # 10 not tested
    add("## 10. Not tested")
    add("")
    if report["not_tested"]:
        add("| Item | Status | Reason |")
        add("| --- | --- | --- |")
        for item in report["not_tested"]:
            add(f"| {item['item']} | {item['status']} | {item['reason']} |")
    else:
        add("Every optimization selected for this experiment was applicable and was "
            "measured; nothing was skipped.")
    add("")

    # 11 raw outputs
    add("## 11. Representative raw outputs")
    add("")
    for block in report["raw_outputs"]:
        add(f"### {block['label']}")
        add("")
        add(f"Options: `{block['options']}`")
        add("")
        for entry in block["runs"]:
            metrics = entry["metrics"]
            add(f"**Run #{entry['run_index']}** - status {entry['status']}, "
                f"quality {entry['quality'] if entry['quality'] is not None else 'N/A'}, "
                f"latency {_fmt(metrics.get('wall_seconds'), 2, ' s')}, "
                f"{metrics.get('output_tokens') or 'N/A'} tokens")
            add("")
            if entry["error"]:
                add(f"Error: {entry['error']}")
                add("")
            else:
                add("```text")
                add(entry["output"] or "(empty output)")
                add("```")
                add("")

    # 12 limitations
    add("## 12. Limitations")
    add("")
    for item in report["limitations"]:
        add(f"**{item['title']}.** {item['text']}")
        add("")

    return "\n".join(lines)


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def _gpu_text(gpu: Any) -> str:
    if not gpu:
        return "none detected (nvidia-smi unavailable)"
    if isinstance(gpu, list):
        return ", ".join(f"{g.get('name')} ({g.get('memory_total_mb')} MB)" for g in gpu)
    return str(gpu)


def write_markdown(report: Dict[str, Any]) -> Path:
    settings = get_settings()
    directory = Path(settings.report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report['meta']['experiment_id']}.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path
