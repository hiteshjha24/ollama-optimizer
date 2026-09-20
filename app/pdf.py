"""PDF report generation with ReportLab Platypus.

The PDF is laid out as a document, not as dumped HTML: tables are built with
explicit column widths so they cannot overflow the page, long model outputs and
code blocks are hard-wrapped to the text column, headings repeat correctly
across page breaks, table headers repeat on continuation pages, and every page
carries a footer with the page number and the experiment id.
"""

from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)

from .charts import generate_charts
from .config import get_settings
from .logging_setup import get_logger

log = get_logger("pdf")

PAGE_SIZE = A4
MARGIN = 18 * mm
CONTENT_WIDTH = PAGE_SIZE[0] - 2 * MARGIN

INK = colors.HexColor("#1c1c1e")
MUTED = colors.HexColor("#5b6169")
RULE = colors.HexColor("#d7dbe0")
ACCENT = colors.HexColor("#0f766e")
BAND = colors.HexColor("#eef2f5")
CODE_BG = colors.HexColor("#f5f6f8")

CODE_WRAP = 96
OUTPUT_WRAP = 100


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.4,
        leading=13.6, textColor=INK, spaceAfter=6, alignment=TA_LEFT,
    )
    return {
        "title": ParagraphStyle("DocTitle", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=25, leading=29, textColor=INK, alignment=TA_LEFT,
                                spaceAfter=10),
        "subtitle": ParagraphStyle("Subtitle", parent=body, fontSize=12.5, leading=17,
                                   textColor=MUTED, spaceAfter=18),
        "h1": ParagraphStyle("H1", parent=body, fontName="Helvetica-Bold", fontSize=15,
                             leading=19, textColor=INK, spaceBefore=16, spaceAfter=8),
        "h2": ParagraphStyle("H2", parent=body, fontName="Helvetica-Bold", fontSize=11.6,
                             leading=15, textColor=INK, spaceBefore=11, spaceAfter=5),
        "h3": ParagraphStyle("H3", parent=body, fontName="Helvetica-Bold", fontSize=10,
                             leading=13, textColor=ACCENT, spaceBefore=8, spaceAfter=3),
        "body": body,
        "small": ParagraphStyle("Small", parent=body, fontSize=8.2, leading=11.4,
                                textColor=MUTED, spaceAfter=5),
        "bullet": ParagraphStyle("Bullet", parent=body, leftIndent=12, bulletIndent=3,
                                 spaceAfter=3),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=8.2, leading=10.8,
                               spaceAfter=0),
        "cellhead": ParagraphStyle("CellHead", parent=body, fontName="Helvetica-Bold",
                                   fontSize=8.2, leading=10.8, spaceAfter=0,
                                   textColor=colors.white),
        "code": ParagraphStyle("Code", parent=body, fontName="Courier", fontSize=7.4,
                               leading=9.6, textColor=INK, spaceAfter=0),
        "quote": ParagraphStyle("Quote", parent=body, fontSize=9, leading=13,
                                leftIndent=10, textColor=MUTED, spaceBefore=4,
                                spaceAfter=8, borderPadding=0),
        "footer": ParagraphStyle("Footer", parent=body, fontSize=7.6, leading=9,
                                 textColor=MUTED),
    }


def esc(text: Any) -> str:
    return html.escape(str(text if text is not None else ""), quote=False)


class _Doc(BaseDocTemplate):
    """Document template that stamps a footer with the page number."""

    def __init__(self, path: str, experiment_id: str, model: str, **kwargs: Any):
        super().__init__(path, pagesize=PAGE_SIZE, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN, bottomMargin=MARGIN + 6 * mm, **kwargs)
        self.experiment_id = experiment_id
        self.model = model
        frame = Frame(MARGIN, MARGIN + 6 * mm, CONTENT_WIDTH,
                      PAGE_SIZE[1] - 2 * MARGIN - 6 * mm, id="body")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=self._cover_page),
            PageTemplate(id="main", frames=[frame], onPage=self._decorate),
        ])

    def _cover_page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFillColor(ACCENT)
        canvas.rect(MARGIN, PAGE_SIZE[1] - MARGIN - 4, 62, 4, stroke=0, fill=1)
        canvas.restoreState()

    def _decorate(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        y = MARGIN + 4 * mm
        canvas.line(MARGIN, y + 10, PAGE_SIZE[0] - MARGIN, y + 10)
        canvas.setFont("Helvetica", 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, y, f"{self.model} - experiment {self.experiment_id}")
        canvas.drawRightString(PAGE_SIZE[0] - MARGIN, y, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()


# --------------------------------------------------------------------------
# flowable builders
# --------------------------------------------------------------------------

def _para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _bullets(items: Sequence[str], style: ParagraphStyle) -> List[Any]:
    return [Paragraph(f"\u2022&nbsp;&nbsp;{esc(item)}", style) for item in items]


def _wrap_block(text: str, width: int) -> str:
    """Hard-wrap text to the column so ReportLab never overflows the frame."""
    lines: List[str] = []
    for raw in (text or "").splitlines() or [""]:
        if not raw.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(raw, width=width, replace_whitespace=False,
                                drop_whitespace=False, break_long_words=True,
                                break_on_hyphens=False)
        lines.extend(wrapped or [""])
    return "\n".join(lines[:220])


def _code(text: str, styles: Dict[str, ParagraphStyle], width: int = CODE_WRAP) -> Table:
    content = Preformatted(_wrap_block(text, width), styles["code"])
    table = Table([[content]], colWidths=[CONTENT_WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]],
           widths: Sequence[float], styles: Dict[str, ParagraphStyle],
           align_right: Sequence[int] = ()) -> Table:
    data = [[Paragraph(esc(h), styles["cellhead"]) for h in header]]
    for row in rows:
        data.append([Paragraph(esc(cell), styles["cell"]) for cell in row])
    table = Table(data, colWidths=list(widths), repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
    ]
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    table.setStyle(TableStyle(style))
    return table


def _chart(chart: Optional[Dict[str, Any]], styles: Dict[str, ParagraphStyle]) -> List[Any]:
    if not chart:
        return []
    flow: List[Any] = [_para(esc(chart["title"]), styles["h3"])]
    path = chart.get("path")
    if path and Path(path).exists():
        try:
            image = Image(path)
            ratio = image.imageHeight / float(image.imageWidth)
            image.drawWidth = CONTENT_WIDTH
            image.drawHeight = min(CONTENT_WIDTH * ratio, 108 * mm)
            image.drawWidth = image.drawHeight / ratio
            image.hAlign = "LEFT"
            flow.append(image)
        except Exception as exc:  # pragma: no cover - defensive
            flow.append(_para(f"Chart could not be embedded: {esc(exc)}", styles["small"]))
    else:
        flow.append(_para(esc(chart.get("note", "Chart unavailable.")), styles["small"]))
        return flow
    flow.append(Spacer(1, 3))
    flow.append(_para(esc(chart.get("note", "")), styles["small"]))
    flow.append(Spacer(1, 6))
    return flow


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def generate_pdf(report: Dict[str, Any], out_path: Optional[Path] = None) -> Tuple[Path, int]:
    settings = get_settings()
    directory = Path(settings.report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(out_path) if out_path else directory / f"{report['meta']['experiment_id']}.pdf"

    styles = _styles()
    charts = generate_charts(report)
    story: List[Any] = []

    story += _cover(report, styles)
    story.append(NextPageTemplate("main"))
    story.append(PageBreak())
    story += _executive(report, styles)
    story += _model_section(report, styles)
    story.append(PageBreak())
    story += _methodology(report, styles)
    story += _baseline(report, styles)
    story.append(PageBreak())
    story += _optimizations(report, styles)
    story.append(PageBreak())
    story += _comparison(report, styles, charts)
    story += _tradeoffs(report, styles)
    story.append(PageBreak())
    story += _recommendation(report, styles, charts)
    story += _alternatives(report, styles)
    story += _not_tested(report, styles)
    story.append(PageBreak())
    story += _raw_outputs(report, styles)
    story += _limitations(report, styles)

    doc = _Doc(str(path), report["meta"]["experiment_id"], report["experiment"]["model"],
               title=report["meta"]["title"], author=report["meta"]["app"],
               subject="Local LLM optimization benchmark")
    doc.build(story)
    pages = doc.page
    log.info("PDF written to %s (%s pages)", path, pages)
    return path, pages


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def _cover(report: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    meta, summary = report["meta"], report["executive_summary"]
    info = report["model_info"]
    rec = report["recommended"]
    flow: List[Any] = [Spacer(1, 34 * mm)]
    flow.append(_para("Local model optimization benchmark", st["title"]))
    flow.append(_para(esc(summary["model"]), st["subtitle"]))

    facts = [
        ["Benchmark date", summary["benchmark_date"]],
        ["Prompt", f"{summary['prompt_label']} ({summary['prompt_category']})"],
        ["Optimization scope", summary["strategy"]],
        ["Objective", summary["objective"]],
        ["Configurations tested", str(summary["configurations_tested"])],
        ["Total generations", str(summary["total_runs"])],
        ["Model size / quantization",
         f"{info['parameter_size']} / {info['quantization_level']}"],
        ["Ollama version", meta["ollama_version"]],
        ["Generated by", f"{meta['app']} {meta['app_version']}"],
        ["Experiment ID", meta["experiment_id"]],
    ]
    flow.append(_table(["Field", "Value"], facts, [0.34 * CONTENT_WIDTH, 0.66 * CONTENT_WIDTH], st))
    flow.append(Spacer(1, 10))
    if rec.get("available"):
        flow.append(_para("Recommended configuration", st["h3"]))
        flow.append(_para(esc(rec["label"]), st["body"]))
    flow.append(Spacer(1, 12))
    flow.append(_para(esc(summary.get("caveat", "")), st["quote"]))
    return flow


def _executive(report: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    summary = report["executive_summary"]
    base = summary["baseline"]
    flow = [_para("1. Executive summary", st["h1"])]
    flow.append(_para(
        f"This report benchmarks <b>{esc(summary['model'])}</b> against "
        f"{summary['configurations_tested']} configurations over "
        f"{summary['total_runs']} generations, using the "
        f"<b>{esc(summary['objective'])}</b> weighting. The run took "
        f"{summary['duration_seconds']} seconds.", st["body"]))

    flow.append(_para("Baseline performance", st["h2"]))
    flow.append(_table(
        ["Quality (0-10)", "Median latency", "Tokens/s", "Consistency (0-10)"],
        [[base["quality"], f"{base['median_latency']} s", base["tokens_per_second"],
          base["consistency"]]],
        [CONTENT_WIDTH * 0.25] * 4, st, align_right=(0, 1, 2, 3)))
    flow.append(Spacer(1, 8))

    flow.append(_para("Tested optimization categories", st["h2"]))
    flow.append(_para(esc(", ".join(summary["categories_tested"]) or "none"), st["body"]))

    flow.append(_para("Major findings", st["h2"]))
    flow.extend(_bullets(summary["major_findings"], st["bullet"]))
    flow.append(Spacer(1, 6))

    rec = report["recommended"]
    if rec.get("available"):
        flow.append(_para("Recommended configuration", st["h2"]))
        flow.append(_para(f"<b>{esc(rec['label'])}</b>", st["body"]))
        flow.append(_code(json.dumps(rec["options"], indent=2), st))
        flow.append(Spacer(1, 6))
    return flow


def _model_section(report: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    info = report["model_info"]
    host = info["host"]
    flow = [_para("2. Model information", st["h1"])]
    rows = [
        ["Model", info["name"]],
        ["Family", info["family"]],
        ["Parameters", info["parameter_size"]],
        ["Quantization", info["quantization_level"]],
        ["Format", info["format"]],
        ["Size on disk", info["size"]],
        ["Digest", info["digest"]],
        ["Last modified", info["modified_at"]],
        ["Ollama version", info["ollama_version"]],
        ["Host platform", host["platform"]],
        ["CPU", f"{host['processor']} ({host['cpu']})"],
        ["System RAM", f"{host['ram_total_gb']} GB" if host["ram_total_gb"] else "not reported"],
        ["GPU", _gpu_text(host["gpu"])],
    ]
    flow.append(_table(["Field", "Value"], rows,
                       [0.30 * CONTENT_WIDTH, 0.70 * CONTENT_WIDTH], st))
    flow.append(Spacer(1, 8))
    flow.append(_para("Quantization", st["h2"]))
    flow.append(_para(esc(info["quantization_note"]), st["body"]))
    flow.extend(_bullets(info["quantization_effects"], st["bullet"]))
    return flow


def _methodology(report: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    method = report["methodology"]
    flow = [_para("3. Methodology", st["h1"])]
    rows = [
        ["Runs per configuration", str(method["runs_per_config"])],
        ["Maximum generated tokens", str(method["max_tokens"])],
        ["Timeout per generation", f"{method['timeout_seconds']} s"],
        ["Streaming", "on" if method["streaming"] else "off"],
        ["Concurrency", f"{method['concurrency']} request(s) at a time"],
        ["Seed (determinism test)", str(method["seed"])],
        ["Evaluator", method["evaluator_mode"]],
        ["Configurations", str(method["configurations"])],
        ["Phases", ", ".join(f"{k}: {v}" for k, v in method["phase_counts"].items())],
    ]
    flow.append(_table(["Setting", "Value"], rows,
                       [0.42 * CONTENT_WIDTH, 0.58 * CONTENT_WIDTH], st))
    flow.append(Spacer(1, 8))

    flow.append(_para("Search strategy", st["h2"]))
    for index, step in enumerate(method["search_strategy"], start=1):
        flow.append(_para(f"{index}.&nbsp;&nbsp;{esc(step)}", st["bullet"]))

    flow.append(_para("Metrics collected", st["h2"]))
    flow.extend(_bullets(method["metrics_collected"], st["bullet"]))

    flow.append(_para("Evaluation methodology", st["h2"]))
    flow.append(_para(esc(method["evaluation_method"]["primary"]), st["body"]))
    criteria = ", ".join(f"{k} ({v})" for k, v in method["evaluation_method"]["criteria"].items())
    flow.append(_para(f"Criteria and weights: {esc(criteria)}.", st["body"]))
    flow.extend(_bullets(method["evaluation_method"]["notes"], st["bullet"]))
    flow.append(_para(f"LLM judge: {esc(method['evaluation_method']['llm_judge'])}", st["small"]))

    flow.append(_para("Test prompt", st["h2"]))
    flow.append(_code(method["prompt"][:3500], st, OUTPUT_WRAP))
    if method.get("system_prompt"):
        flow.append(Spacer(1, 5))
        flow.append(_para("System prompt supplied by the user", st["h3"]))
        flow.append(_code(method["system_prompt"], st, OUTPUT_WRAP))
    return flow


def _baseline(report: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    baseline = report["baseline"]
    flow = [_para("4. Baseline results", st["h1"])]
    if not baseline.get("available"):
        flow.append(_para(esc(baseline.get("reason", "")), st["body"]))
        return flow
    flow.append(_para(esc(baseline["rationale"] or ""), st["body"]))
    flow.append(_para(
        f"{baseline['successful_runs']} of {baseline['runs']} runs succeeded.", st["small"]))
    rows = [[label, value, extra or "-"] for label, value, extra in baseline["metrics"]]
    flow.append(_table(["Measurement", "Value", "Uncertainty / note"], rows,
                       [0.34 * CONTENT_WIDTH, 0.26 * CONTENT_WIDTH, 0.40 * CONTENT_WIDTH],
                       st, align_right=(1,)))
    flow.append(Spacer(1, 8))
    flow.append(_para("Representative baseline output", st["h2"]))
    flow.append(_code(baseline["sample_output"], st, OUTPUT_WRAP))
    return flow


def _optimizations(report: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    flow = [_para("5. Optimization results", st["h1"])]
    sections = report["optimization_results"]
    if not sections:
        flow.append(_para("No optimization categories were run in this experiment.", st["body"]))
        return flow

    for index, section in enumerate(sections, start=1):
        block: List[Any] = [_para(f"5.{index} {esc(section['name'])}", st["h2"])]
        block.append(_para(f"<b>What changed:</b> {esc(section['what_changed'])}", st["body"]))
        block.append(_para(f"<b>Why it was tested:</b> {esc(section['why_tested'])}", st["body"]))
        flow.append(KeepTogether(block))

        rows = [[cfg["label"], cfg["phase"], cfg["quality"], cfg["latency_median"],
                 cfg["tokens_per_second"], cfg["consistency"], cfg["failure_rate"]]
                for cfg in section["configurations"]]
        widths = [0.30, 0.11, 0.11, 0.15, 0.12, 0.12, 0.09]
        flow.append(_table(
            ["Configuration", "Phase", "Quality", "Median latency", "Tokens/s",
             "Consistency", "Failures"],
            rows, [w * CONTENT_WIDTH for w in widths], st, align_right=(2, 3, 4, 5, 6)))
        flow.append(Spacer(1, 5))
        flow.append(_para(esc(section["summary"]), st["body"]))

        for cfg in section["configurations"][:3]:
            if cfg.get("rationale"):
                flow.append(_para(f"<b>{esc(cfg['label'])}</b> - {esc(cfg['rationale'])}",
                                  st["small"]))
        # show the rewritten prompt where one exists
        changed = next((c for c in section["configurations"] if c.get("prompt_changed")), None)
        if changed and changed.get("prompt"):
            flow.append(_para(f"Prompt variant used by '{esc(changed['label'])}'", st["h3"]))
            flow.append(_code(changed["prompt"][:1400], st, OUTPUT_WRAP))
        sys_cfg = next((c for c in section["configurations"] if c.get("system_prompt")), None)
        if sys_cfg:
            flow.append(_para(f"System prompt used by '{esc(sys_cfg['label'])}'", st["h3"]))
            flow.append(_code(sys_cfg["system_prompt"][:900], st, OUTPUT_WRAP))
        flow.append(Spacer(1, 8))
    return flow


def _comparison(report: Dict[str, Any], st: Dict[str, ParagraphStyle],
                charts: Dict[str, Dict[str, Any]]) -> List[Any]:
    comparison = report["comparison"]
    flow = [_para("6. Comparative analysis", st["h1"])]
    rows = [[row["rank"], row["label"], row["quality"], row["speed_score"],
             row["consistency"], row["avg_latency"], row["tokens_per_second"],
             row["composite"]] for row in comparison["rows"]]
    widths = [0.05, 0.31, 0.10, 0.10, 0.12, 0.11, 0.10, 0.11]
    flow.append(_table(
        ["#", "Configuration", "Quality", "Speed", "Consistency", "Avg latency",
         "Tokens/s", "Composite"],
        rows, [w * CONTENT_WIDTH for w in widths], st, align_right=(2, 3, 4, 5, 6, 7)))
    flow.append(Spacer(1, 5))
    flow.append(_para(esc(comparison["note"]), st["small"]))
    flow.append(Spacer(1, 8))

    for key in ("quality", "latency", "throughput"):
        flow += _chart(charts.get(key), st)
    flow.append(PageBreak())
    for key in ("quality_speed", "distribution", "consistency"):
        flow += _chart(charts.get(key), st)
    return flow


def _tradeoffs(report: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    flow = [_para("7. Optimization trade-offs", st["h1"])]
    tradeoffs = report["tradeoffs"]
    if not tradeoffs:
        flow.append(_para(
            "No configuration differed from the baseline by more than one percent on any "
            "tracked metric, so there are no trade-offs to report.", st["body"]))
        return flow
    for item in tradeoffs:
        block = [_para(esc(item["label"]), st["h3"])]
        for gain in item["gains"]:
            block.append(_para(f"+&nbsp;&nbsp;{esc(gain)}", st["bullet"]))
        for cost in item["costs"]:
            block.append(_para(f"-&nbsp;&nbsp;{esc(cost)}", st["bullet"]))
        flow.append(KeepTogether(block))
        flow.append(Spacer(1, 4))
    return flow


def _recommendation(report: Dict[str, Any], st: Dict[str, ParagraphStyle],
                    charts: Dict[str, Dict[str, Any]]) -> List[Any]:
    rec = report["recommended"]
    flow = [_para("8. Recommended configuration", st["h1"])]
    if not rec.get("available"):
        flow.append(_para(esc(rec.get("reason", "")), st["body"]))
        return flow

    weights = ", ".join(f"{k}: {v}" for k, v in rec["weights"].items())
    flow.append(_table(
        ["Field", "Value"],
        [["Model", rec["model"]], ["Objective", rec["objective"]],
         ["Weights", weights], ["Configuration", rec["label"]]],
        [0.26 * CONTENT_WIDTH, 0.74 * CONTENT_WIDTH], st))
    flow.append(Spacer(1, 8))

    flow.append(_para("Configuration", st["h2"]))
    flow.append(_code(json.dumps(rec["options"], indent=2), st))
    if rec.get("system_prompt"):
        flow.append(Spacer(1, 4))
        flow.append(_para("System prompt", st["h3"]))
        flow.append(_code(rec["system_prompt"], st, OUTPUT_WRAP))
    flow.append(Spacer(1, 8))

    observed = rec["observed"]
    flow.append(_para("Observed results", st["h2"]))
    flow.append(_table(
        ["Quality", "Tokens/s", "Median latency", "Consistency", "Runs"],
        [[f"{observed['quality']}/10", observed["tokens_per_second"],
          observed["median_latency"], f"{observed['consistency']}/10", observed["runs"]]],
        [CONTENT_WIDTH * 0.2] * 5, st, align_right=(0, 1, 2, 3, 4)))
    if observed.get("quality_ci"):
        flow.append(_para(esc(observed["quality_ci"]), st["small"]))
    flow.append(Spacer(1, 6))

    comparison = rec.get("vs_baseline") or {}
    if comparison.get("available"):
        flow.append(_para("Compared with baseline", st["h2"]))
        flow.append(_table(
            ["Quality", "Tokens/s", "Median latency", "Consistency"],
            [[_pct(comparison.get("quality_percent")),
              _pct(comparison.get("tokens_per_second_percent")),
              _pct(comparison.get("median_latency_percent")),
              _pct(comparison.get("consistency_percent"))]],
            [CONTENT_WIDTH * 0.25] * 4, st, align_right=(0, 1, 2, 3)))
        flow.append(Spacer(1, 4))
        significance = comparison.get("quality_significance") or {}
        if significance.get("note"):
            flow.append(_para(esc(significance["note"]), st["small"]))
    flow.append(Spacer(1, 6))

    flow.append(_para("Why this configuration was selected", st["h2"]))
    flow.append(_para(esc(rec["why"]), st["body"]))
    flow.append(_para(esc(rec["caveat"]), st["quote"]))

    flow += _chart(charts.get("baseline_vs_best"), st)

    flow.append(_para("How to apply it", st["h2"]))
    flow.append(_code(rec["how_to_apply"]["api"], st))
    flow.append(Spacer(1, 4))
    flow.append(_para(esc(rec["reversibility"]), st["small"]))
    return flow


def _alternatives(report: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    flow = [_para("9. Alternative configurations", st["h1"])]
    alternatives = report["alternatives"]
    if not alternatives:
        flow.append(_para(
            "The recommended configuration also led on every individual metric, so no "
            "separate quality-first or speed-first alternative is offered.", st["body"]))
        return flow
    rows = []
    for name, alt in alternatives.items():
        comparison = alt.get("comparison") or {}
        rows.append([name.replace("_", " ").title(), alt["label"],
                     f"{alt['composite_score']:.2f}" if alt.get("composite_score") is not None else "N/A",
                     comparison.get("interpretation", alt.get("basis", ""))])
    flow.append(_table(["Objective", "Configuration", "Composite", "Observed difference"],
                       rows, [0.18 * CONTENT_WIDTH, 0.25 * CONTENT_WIDTH,
                              0.11 * CONTENT_WIDTH, 0.46 * CONTENT_WIDTH], st))
    return flow


def _not_tested(report: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    items = report.get("not_tested") or []
    if not items:
        return []
    flow = [_para("10. Not tested", st["h1"])]
    flow.append(_para(
        "These techniques were considered but not executed. They are listed so the "
        "report cannot be mistaken for a complete sweep of everything Ollama exposes.",
        st["body"]))
    rows = [[item["item"], item["status"], item["reason"]] for item in items]
    flow.append(_table(["Item", "Status", "Reason"], rows,
                       [0.26 * CONTENT_WIDTH, 0.18 * CONTENT_WIDTH, 0.56 * CONTENT_WIDTH], st))
    return flow


def _raw_outputs(report: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    flow = [_para("11. Representative raw outputs", st["h1"])]
    flow.append(_para(
        "Every conclusion above rests on text the model actually produced. A sample is "
        "reproduced here; the application stores every run in full.", st["body"]))
    for block in report["raw_outputs"]:
        flow.append(_para(esc(block["label"]), st["h2"]))
        flow.append(_para(f"Options: {esc(json.dumps(block['options']))}", st["small"]))
        for entry in block["runs"]:
            metrics = entry["metrics"]
            header = (f"Run #{entry['run_index']} - {entry['status']}, quality "
                      f"{entry['quality'] if entry['quality'] is not None else 'N/A'}, "
                      f"latency {_num(metrics.get('wall_seconds'))} s, "
                      f"{metrics.get('output_tokens') if metrics.get('output_tokens') is not None else 'N/A'} tokens, "
                      f"{_num(metrics.get('tokens_per_second'), 1)} tok/s")
            flow.append(_para(esc(header), st["h3"]))
            if entry["error"]:
                flow.append(_para(f"Error: {esc(entry['error'])}", st["small"]))
                continue
            flow.append(_code(entry["output"] or "(empty output)", st, OUTPUT_WRAP))
            criteria = entry.get("criteria") or {}
            if criteria:
                bits = []
                for name, data in criteria.items():
                    score = data.get("score")
                    bits.append(f"{name}: {score if score is not None else 'not scorable'}")
                flow.append(_para(esc("Evaluation - " + "; ".join(bits)), st["small"]))
            flow.append(Spacer(1, 4))
    return flow


def _limitations(report: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    flow = [_para("12. Limitations", st["h1"])]
    for item in report["limitations"]:
        block = [_para(esc(item["title"]), st["h3"]),
                 _para(esc(item["text"]), st["body"])]
        flow.append(KeepTogether(block))
    return flow


def _pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:+.1f}%"


def _num(value: Optional[float], digits: int = 2) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{value:.{digits}f}"


def _gpu_text(gpu: Any) -> str:
    if not gpu:
        return "none detected (nvidia-smi unavailable)"
    if isinstance(gpu, list):
        return ", ".join(f"{g.get('name')} ({g.get('memory_total_mb')} MB)" for g in gpu)
    return str(gpu)
