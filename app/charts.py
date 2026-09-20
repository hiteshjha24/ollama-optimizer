"""Charts, rendered from the benchmark's own data.

If matplotlib is unavailable the report still builds; each chart slot simply
reports that it could not be rendered, and the PDF prints the explanation
instead of the image.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .logging_setup import get_logger

log = get_logger("charts")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

INK = "#1c1c1e"
BASE_COLOR = "#6b7280"
OPT_COLOR = "#2563eb"
WIN_COLOR = "#0f766e"
GRID = "#e5e7eb"
MAX_BARS = 14


def _short(label: str, width: int = 26) -> str:
    label = (label or "").replace(" (model defaults)", "")
    return label if len(label) <= width else label[: width - 1] + "\u2026"


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def _pick(rows: List[Dict[str, Any]], limit: int = MAX_BARS) -> List[Dict[str, Any]]:
    """Baseline plus the highest-ranked configurations, keeping the chart legible."""
    baseline = [r for r in rows if r.get("is_baseline")]
    others = [r for r in rows if not r.get("is_baseline")]
    return (baseline + others)[:limit]


def generate_charts(report: Dict[str, Any], out_dir: Optional[Path] = None
                    ) -> Dict[str, Dict[str, Any]]:
    """Return ``{chart_id: {"path": str|None, "title": str, "note": str}}``."""
    charts: Dict[str, Dict[str, Any]] = {}
    if not HAVE_MPL:
        note = ("matplotlib is not installed, so charts could not be rendered. "
                "All underlying numbers remain in the tables.")
        for key, title in (("quality", "Quality by configuration"),
                           ("latency", "Median latency by configuration"),
                           ("throughput", "Tokens per second by configuration"),
                           ("quality_speed", "Quality versus speed"),
                           ("baseline_vs_best", "Baseline versus recommended"),
                           ("consistency", "Quality across repeated runs"),
                           ("distribution", "Latency distribution")):
            charts[key] = {"path": None, "title": title, "note": note}
        return charts

    directory = out_dir or Path(tempfile.mkdtemp(prefix="ollama-opt-charts-"))
    directory.mkdir(parents=True, exist_ok=True)

    ranking = report["recommendation"].get("ranking", [])
    configs = {c["id"]: c for c in report["configs"]}
    rows = _pick(ranking)

    charts["quality"] = _bar_chart(
        directory / "quality.png", rows, "quality", "Quality score (0-10)",
        "Quality by configuration", digits=2,
        error_key="ci95",
    )
    charts["latency"] = _bar_chart(
        directory / "latency.png", rows, "latency_median", "Median latency (s)",
        "Median latency by configuration", digits=2, lower_is_better=True,
    )
    charts["throughput"] = _bar_chart(
        directory / "throughput.png", rows, "tokens_per_second", "Tokens per second",
        "Generation throughput by configuration", digits=1,
    )
    charts["quality_speed"] = _scatter(directory / "quality_speed.png", rows)
    charts["baseline_vs_best"] = _baseline_vs_best(directory / "baseline_vs_best.png",
                                                   report, ranking)
    charts["consistency"] = _consistency(directory / "consistency.png", report, ranking)
    charts["distribution"] = _distribution(directory / "distribution.png", rows)
    return charts


def _values(rows: List[Dict[str, Any]], metric: str) -> Tuple[List[str], List[float],
                                                              List[bool], List[Optional[List[float]]]]:
    labels, values, is_base, cis = [], [], [], []
    for row in rows:
        agg = row.get("aggregate", {})
        if metric == "quality":
            block = agg.get("quality") or {}
            value, ci = block.get("mean"), block.get("ci95")
        elif metric == "latency_median":
            block = agg.get("latency") or {}
            value, ci = block.get("median"), None
        elif metric == "tokens_per_second":
            block = agg.get("tokens_per_second") or {}
            value, ci = block.get("mean"), block.get("ci95")
        else:
            value, ci = None, None
        if value is None:
            continue
        labels.append(_short(row.get("label", "")))
        values.append(float(value))
        is_base.append(bool(row.get("is_baseline")))
        cis.append(ci)
    return labels, values, is_base, cis


def _bar_chart(path: Path, rows: List[Dict[str, Any]], metric: str, ylabel: str,
               title: str, digits: int = 2, lower_is_better: bool = False,
               error_key: Optional[str] = None) -> Dict[str, Any]:
    labels, values, is_base, cis = _values(rows, metric)
    if not values:
        return {"path": None, "title": title,
                "note": "No configuration reported this metric, so the chart is empty."}

    best_index = (values.index(min(values)) if lower_is_better else values.index(max(values)))
    colors = [BASE_COLOR if base else OPT_COLOR for base in is_base]
    colors[best_index] = WIN_COLOR

    errors = None
    if error_key:
        lows, highs = [], []
        for value, ci in zip(values, cis):
            if ci:
                lows.append(max(0.0, value - ci[0]))
                highs.append(max(0.0, ci[1] - value))
            else:
                lows.append(0.0)
                highs.append(0.0)
        if any(lows) or any(highs):
            errors = [lows, highs]

    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=170)
    bars = ax.bar(range(len(values)), values, color=colors, width=0.68)
    if errors:
        ax.errorbar(range(len(values)), values, yerr=errors, fmt="none",
                    ecolor="#9ca3af", elinewidth=1, capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=34, ha="right")
    _style(ax, title, "", ylabel)
    span = (max(values) - min(values)) or max(values) or 1
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + span * 0.03,
                f"{value:.{digits}f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_ylim(0, max(values) + span * 0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    note = ("Grey = baseline, blue = tested configuration, teal = best on this metric."
            + (" Error bars show the 95% confidence interval where one could be computed."
               if errors else ""))
    return {"path": str(path), "title": title, "note": note}


def _scatter(path: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    title = "Quality versus speed"
    points = []
    for row in rows:
        agg = row.get("aggregate", {})
        quality = (agg.get("quality") or {}).get("mean")
        tps = (agg.get("tokens_per_second") or {}).get("mean")
        if quality is None or tps is None:
            continue
        points.append((tps, quality, _short(row.get("label", ""), 18),
                       bool(row.get("is_baseline"))))
    if len(points) < 2:
        return {"path": None, "title": title,
                "note": "Fewer than two configurations reported both quality and throughput."}

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=170)
    for tps, quality, label, base in points:
        ax.scatter(tps, quality, s=70 if base else 52,
                   color=BASE_COLOR if base else OPT_COLOR,
                   marker="s" if base else "o", zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(label, (tps, quality), textcoords="offset points", xytext=(6, 4),
                    fontsize=6.5, color=INK)
    _style(ax, title, "Tokens per second (higher is better)", "Quality score (0-10)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path), "title": title,
            "note": ("Top-right is the desirable corner: fast and high scoring. The square "
                     "marker is the baseline.")}


def _baseline_vs_best(path: Path, report: Dict[str, Any],
                      ranking: List[Dict[str, Any]]) -> Dict[str, Any]:
    title = "Baseline versus recommended configuration"
    baseline = next((r for r in ranking if r["is_baseline"]), None)
    winner = report["recommendation"].get("recommended")
    if not baseline or not winner or winner["configuration_id"] == baseline["configuration_id"]:
        return {"path": None, "title": title,
                "note": ("The baseline is itself the recommended configuration, so there is "
                         "nothing to compare side by side.")}

    metrics = [
        ("Quality\n(0-10)", ("quality", "mean"), 1.0),
        ("Tokens/s", ("tokens_per_second", "mean"), 1.0),
        ("Median\nlatency (s)", ("latency", "median"), 1.0),
        ("Consistency\n(0-10)", ("consistency", "score"), 1.0),
    ]
    labels, base_values, win_values = [], [], []
    for label, (block, key), _ in metrics:
        base_block = (baseline.get("aggregate", {}) or {}).get(block) or {}
        win_block = (winner.get("aggregate", {}) or {}).get(block) or {}
        base_value, win_value = base_block.get(key), win_block.get(key)
        if base_value is None or win_value is None:
            continue
        labels.append(label)
        base_values.append(float(base_value))
        win_values.append(float(win_value))
    if not labels:
        return {"path": None, "title": title,
                "note": "No metric was available for both configurations."}

    # normalise each pair against the baseline so different units share an axis
    base_norm = [100.0] * len(labels)
    win_norm = [(w / b * 100.0) if b else 0.0 for w, b in zip(win_values, base_values)]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=170)
    ax.bar([i - 0.19 for i in x], base_norm, width=0.36, color=BASE_COLOR, label="Baseline")
    ax.bar([i + 0.19 for i in x], win_norm, width=0.36, color=WIN_COLOR,
           label=_short(winner["label"], 24))
    ax.axhline(100, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    _style(ax, title, "", "Relative to baseline (baseline = 100)")
    for i, (bv, wv, raw_b, raw_w) in enumerate(zip(base_norm, win_norm, base_values, win_values)):
        ax.text(i - 0.19, bv + 2, f"{raw_b:.2f}", ha="center", fontsize=6.5, color=INK)
        ax.text(i + 0.19, wv + 2, f"{raw_w:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2)
    ax.set_ylim(0, max(base_norm + win_norm) * 1.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path), "title": title,
            "note": ("Bars are scaled so the baseline is 100 on every metric; the raw value "
                     "is printed above each bar. For latency, lower is better, so a bar "
                     "below 100 is an improvement.")}


def _consistency(path: Path, report: Dict[str, Any],
                 ranking: List[Dict[str, Any]]) -> Dict[str, Any]:
    title = "Quality across repeated runs"
    series = []
    for row in ranking[:4]:
        runs = report["runs_by_config"].get(row["configuration_id"], [])
        scores = [(r["run_index"] + 1, (r.get("evaluation") or {}).get("quality_score"))
                  for r in runs if r.get("status") == "ok"]
        scores = [(i, s) for i, s in scores if isinstance(s, (int, float))]
        if len(scores) >= 2:
            series.append((_short(row["label"], 22), scores, bool(row["is_baseline"])))
    if not series:
        return {"path": None, "title": title,
                "note": "No configuration had two or more scorable runs to plot."}

    fig, ax = plt.subplots(figsize=(8.0, 3.9), dpi=170)
    palette = [OPT_COLOR, WIN_COLOR, "#b45309", "#7c3aed"]
    for index, (label, scores, base) in enumerate(series):
        xs = [s[0] for s in scores]
        ys = [s[1] for s in scores]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6,
                color=BASE_COLOR if base else palette[index % len(palette)], label=label)
    ax.set_xticks(sorted({x for _, scores, _ in series for x, _ in scores}))
    _style(ax, title, "Run number", "Quality score (0-10)")
    ax.set_ylim(0, 10.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path), "title": title,
            "note": ("A flat line means the configuration scored the same on every repeat; "
                     "a jagged line means run-to-run variation that a single test would "
                     "have hidden.")}


def _distribution(path: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    title = "Latency distribution per configuration"
    data, labels = [], []
    for row in rows[:10]:
        samples = (row.get("aggregate", {}).get("samples") or {}).get("latency") or []
        samples = [float(s) for s in samples if isinstance(s, (int, float))]
        if len(samples) >= 2:
            data.append(samples)
            labels.append(_short(row.get("label", ""), 18))
    if not data:
        return {"path": None, "title": title,
                "note": "Not enough per-run latency samples to draw a distribution."}

    fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=170)
    box = ax.boxplot(data, patch_artist=True, widths=0.55,
                     medianprops={"color": INK, "linewidth": 1.2})
    for patch in box["boxes"]:
        patch.set_facecolor("#dbeafe")
        patch.set_edgecolor(OPT_COLOR)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=34, ha="right")
    _style(ax, title, "", "Wall-clock latency (s)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path), "title": title,
            "note": ("Each box spans the interquartile range of that configuration's runs, "
                     "with the median marked. Wide boxes mean unstable latency.")}
