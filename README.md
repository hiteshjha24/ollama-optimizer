# Ollama Optimizer

A local web application that benchmarks a model served by your own Ollama
instance, systematically searches for a better configuration, and writes up the
result as a comparison dashboard, a Markdown report and a paginated PDF.

Everything it reports is measured. Anything it could not measure is labelled
`N/A` or "Not tested" **with the reason**, never estimated and never filled in
with a plausible-looking number.

---

## Quick start

```bash
# 1. make sure Ollama is running and has at least one model
ollama serve
ollama pull llama3.1:8b

# 2. start the app (from the project root)
python3 run.py

# 3. open the UI
#    http://127.0.0.1:8848
```

Then: **Models** → pick one → **New Experiment** → choose a prompt →
**Start Full Optimization**.

Other options:

```bash
python3 run.py --port 9000        # different port
python3 run.py --open             # open a browser window too
python3 run.py --host 0.0.0.0     # expose on the LAN (see Security below)
python3 run.py --log-level DEBUG
```

Run the tests (no Ollama required — they use a mock server):

```bash
python3 tests/run_tests.py
python3 tests/run_tests.py -v               # verbose
python3 tests/run_tests.py test_pipeline    # one module
python3 tests/run_tests.py --integration    # also test against a real Ollama
```

---

## Requirements

- **Python 3.10+** (developed and tested on 3.12)
- **Ollama** running locally with at least one model pulled
- Optional but recommended: `reportlab` (PDF), `matplotlib` (charts in the PDF),
  `psutil` (CPU/RAM sampling)

```bash
pip install -r requirements.txt
```

The app degrades gracefully without the optional packages: without `reportlab`
you still get the Markdown report and the PDF step reports why it was skipped;
without `matplotlib` the PDF is generated without charts and says so; without
`psutil` resource columns read "N/A — metric unavailable on this system".

### Dependency choices (deliberate)

The backend uses **only the Python standard library** — `http.server`,
`sqlite3`, `urllib`, `threading`, `unittest` — instead of FastAPI/uvicorn/
SQLAlchemy/httpx/pytest, and the frontend is **vanilla HTML/CSS/JS** with no
build step, no npm and no CDN.

Why: this application was built and validated in an offline environment where
third-party packages could not be installed. Rather than ship untested code that
imports FastAPI, the whole stack was written against what could actually be
executed and tested. The practical consequences are worth knowing:

- `python3 run.py` is the only start command — there is no `uvicorn` process, no
  `npm install`, no bundler, and the app works with no network access at all.
- The server is a threaded `http.server`. It is intended for **local, single-user
  use**, which is what benchmarking a local model is. It is not hardened for
  public exposure.
- Validation is hand-rolled in the API layer rather than done by pydantic.
- Tests use `unittest` rather than `pytest` (the runner takes the same kinds of
  arguments: see `tests/run_tests.py --help`).

If you want to port it, the seams are clean: `app/server.py` is the only file
that knows about HTTP, and `app/db.py` is the only file that knows about SQL.

---

## What it does

### 1. Discovery
Lists models from `/api/tags`, enriches them with `/api/show` (parameter size,
quantization, family, format, declared context length, Modelfile system prompt),
and reports loaded models from `/api/ps`. If Ollama is unreachable the UI says
so and explains how to start it — it never shows an empty list as if there were
no models.

### 2. Experiment setup
Pick a model, write a prompt or choose one of **8 built-in benchmark prompts**
(reasoning, coding, summarization, factual QA, creative, instruction-following,
structured JSON, long-context lookup). The built-in prompts carry machine-
checkable expectations, which makes correctness and format scoring much stronger
than it can be for free text.

You control runs per configuration (default **5**), max tokens, timeout,
streaming, concurrency (default **1**), seed, evaluation mode, and the
recommendation objective.

### 3. One-click optimization pipeline
**Start Full Optimization** runs, in order:

1. **Discovery** — model capabilities and host info.
2. **Planning** — each optimization declares whether it applies; those that do
   not are recorded with a reason.
3. **Warm-up** — one unmeasured generation so the first measured run does not
   include model load time.
4. **Baseline** — the model's own defaults, with only an output-token cap.
5. **Coarse search** — one factor at a time (OFAT) per optimization family.
6. **Fine search** — the best value of each factor is combined, and the most
   influential factor is probed at neighbouring values.
7. **Validation** — the top configurations are re-sampled with extra runs.
8. **Analysis → Markdown report → PDF.**

Coarse→fine keeps the cost linear rather than multiplicative: a full sweep of
4 sampling parameters is ~17 configurations, not the ~108 a Cartesian product
would need.

### Optimization families

| Family | What it varies | Notes |
| --- | --- | --- |
| Generation parameters | `temperature`, `top_p`, `top_k`, `repeat_penalty`, fixed `seed` | OFAT then combined |
| Prompt engineering | structured, checklist, role, decomposed, condensed, format-only rewrites | the task is preserved |
| System prompt | minimal / accuracy / concise / format-focused variants | user prompt untouched |
| Context | `num_ctx` candidates bounded by the model's declared context, plus trimmed and question-first variants for long prompts | never exceeds declared context |
| Runtime | non-streaming, `num_thread`, `num_batch` | GPU layer count, `keep_alive=0`, `OLLAMA_NUM_PARALLEL` and concurrent throughput are reported as **Not tested**, with reasons |
| Quantization | compares against another **locally installed** variant of the same family | if no other variant exists it is informational only, never simulated |

Only options documented by the Ollama API are ever sent; anything else is
dropped before the request.

### 4. Measurement
Per run: wall-clock latency, time-to-first-token (streaming only — it is left
empty, not estimated, when streaming is off), Ollama's `total_duration`,
`load_duration`, `prompt_eval_duration`, `eval_duration`, prompt and output
token counts, tokens/second, truncation flag, and system CPU/RAM during the run.

Per configuration: n, mean, median, min, max, standard deviation, coefficient of
variation and a 95% confidence interval (only when n ≥ 3). Baseline comparisons
carry a Welch's t-test flag, explicitly labelled as indicative on small samples.

### 5. Quality evaluation
A deterministic heuristic evaluator (`heuristic-v1`) scores six criteria —
correctness (28%), instruction following (22%), completeness (16%), format
compliance (16%), relevance (10%), coherence (8%) — renormalised over whatever
is actually scorable. Criteria that cannot be checked for a given prompt are
marked "not scorable" rather than guessed; for example, correctness is not
scored for an open-ended creative prompt. Consistency across runs combines
pairwise content similarity with quality-score stability.

An optional LLM-as-judge pass can be enabled; its scores are stored and shown
separately and always labelled as model estimates.

### 6. Recommendation
Component scores (quality, speed, consistency, efficiency) are min-max
normalised across the experiment to 0–10, then combined with the objective's
weights:

| Objective | Quality | Speed | Consistency | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| Quality first | 60% | 25% | 15% | — |
| Speed first | 20% | 60% | 10% | 10% |
| Balanced | 40% | 30% | 30% | — |
| Custom | you choose | | | |

The report shows the full arithmetic: each component's score, its weight, its
contribution, and the sum. Configurations with failed runs are penalised in
proportion to their failure rate. The baseline can and does win when nothing
beats it. Alternatives (maximum quality / maximum speed / most consistent) are
listed separately, and every recommendation is framed conditionally: it is the
best under *these* weights, *this* prompt and *this* hardware.

### 7. Reports
- **Markdown** — 12 numbered sections: executive summary, model information,
  methodology, baseline, per-family optimization results, comparative analysis,
  trade-offs, recommendation (with a ready-to-paste `curl` command),
  alternatives, not tested, representative raw outputs, limitations.
- **PDF** — a paginated A4 document with a cover page, page numbers, tables and
  charts (quality by configuration, latency, throughput, quality-vs-speed
  scatter, baseline-vs-best, consistency, distribution). A small experiment
  produces roughly 8–13 pages; a full-strategy run produces about 20.

Both are downloadable from the experiment page and the Reports page, and can be
regenerated at any time.

---

## Safety and reversibility

- **Nothing on your machine is modified.** Every configuration is sent as
  request-time options on `/api/generate`. No Modelfile is written, no model is
  created, copied or deleted, no Ollama setting is changed. There is nothing to
  undo, and the recommendation section says exactly how to apply the winning
  settings from your own client.
- The app only ever calls documented read/generate endpoints.
- Concurrency defaults to 1, because local inference is resource-bound and
  parallel requests distort latency measurements.
- Cancellation is honoured between and during runs; work already completed is
  kept and remains in the database.
- A single failed generation is recorded with its error and the experiment
  continues; the run only aborts after repeated fatal errors (model removed
  mid-experiment, Ollama gone), and then says why.

### Security

The server binds to `127.0.0.1` by default and has no authentication. `--host
0.0.0.0` exposes an interface that can start generations on your machine and
read your experiment history — only do that on a network you trust.

---

## Project structure

```
ollama-optimizer/
├── run.py                     # entry point: python3 run.py
├── requirements.txt           # optional-but-recommended packages
├── .env.example               # all supported environment variables
├── README.md
├── app/
│   ├── version.py             # app name / version / schema version
│   ├── config.py              # settings: data/settings.json > env/.env > defaults
│   ├── logging_setup.py       # rotating file + console logging
│   ├── db.py                  # SQLite schema and all queries (WAL, per-thread conns)
│   ├── ollama.py              # Ollama client: discovery, generate/chat, errors, metrics
│   ├── metrics.py             # summary statistics, Welch's t, resource sampling, host info
│   ├── prompts.py             # 8 benchmark prompts with checkable expectations
│   ├── evaluation.py          # deterministic quality evaluator + optional LLM judge
│   ├── optimizations/
│   │   ├── base.py            # plugin interface: Applicability, RunConfig, Optimization
│   │   ├── generation.py      # baseline + sampling-parameter search
│   │   ├── prompting.py       # prompt and system-prompt variants
│   │   ├── context.py         # num_ctx and context-shape variants
│   │   ├── runtime.py         # streaming/threads/batch + explicit "not tested" entries
│   │   ├── quantization.py    # cross-variant comparison, informational otherwise
│   │   └── __init__.py        # registry and strategy sets
│   ├── recommend.py           # normalisation, weighting, ranking, trade-offs
│   ├── engine.py              # experiment manager, pipeline, progress, cancellation
│   ├── report.py              # structured report + Markdown renderer
│   ├── charts.py              # matplotlib charts (optional dependency)
│   ├── pdf.py                 # ReportLab paginated PDF (optional dependency)
│   ├── server.py              # stdlib HTTP API, SSE, static files
│   └── static/                # index.html, app.js, styles.css (no build step)
├── tests/
│   ├── fake_ollama.py         # mock Ollama server (test fixture only)
│   ├── support.py             # sandboxed settings/database per test
│   ├── test_ollama_service.py
│   ├── test_optimizations.py
│   ├── test_evaluation.py
│   ├── test_metrics.py
│   ├── test_recommend.py
│   ├── test_db.py
│   ├── test_pipeline.py       # end-to-end: plan → runs → report → PDF
│   ├── test_api.py            # HTTP layer incl. SSE
│   ├── test_integration.py    # optional, against a real Ollama
│   └── run_tests.py
└── data/                      # created at runtime
    ├── experiments.db
    ├── settings.json
    ├── app.log
    └── reports/               # <experiment_id>.md and <experiment_id>.pdf
```

---

## Configuration

Precedence: **Settings page (`data/settings.json`) > environment / `.env` >
defaults.** See `.env.example` for every variable. The most useful ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | where Ollama listens |
| `APP_HOST` / `APP_PORT` | `127.0.0.1` / `8848` | where this app listens |
| `DEFAULT_RUNS` | `5` | runs per configuration |
| `DEFAULT_MAX_TOKENS` | `512` | `num_predict` cap |
| `DEFAULT_TIMEOUT` | `120` | per-generation timeout (s) |
| `MAX_CONCURRENCY` | `1` | parallel generations |
| `AUTO_PDF` | `true` | render the PDF automatically |
| `EVALUATOR_MODE` | `heuristic` | or `heuristic+llm` |
| `DATABASE_PATH` / `REPORT_DIR` | under `data/` | storage locations |

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | app version, Ollama status, counts |
| GET | `/api/models` | discovered models (+ loaded models) |
| GET | `/api/models/{name}/info` | `/api/show` details and capabilities |
| GET | `/api/prompts` | built-in benchmark prompts |
| GET | `/api/optimizations` | optimization registry, strategies, objectives |
| GET | `/api/dashboard` | dashboard payload |
| GET/PUT | `/api/settings` | read / update editable settings |
| GET/POST | `/api/experiments` | list / start |
| GET/DELETE | `/api/experiments/{id}` | detail (with configurations) / delete |
| POST | `/api/experiments/{id}/cancel` | request cancellation |
| GET | `/api/experiments/{id}/runs` | raw runs (optionally per configuration) |
| GET | `/api/experiments/{id}/progress` | polling fallback for progress |
| GET | `/api/experiments/{id}/events` | **SSE** live progress stream |
| POST | `/api/experiments/{id}/report` | (re)generate Markdown + PDF |
| GET | `/api/experiments/{id}/report` | report as JSON + rendered Markdown |
| GET | `/api/experiments/{id}/report.pdf` | download the PDF |
| GET | `/api/experiments/{id}/report.md` | download the Markdown |
| GET | `/api/reports` | all generated report files |

Errors are always JSON: `{"error": {"status", "message", "detail"}}`.

---

## What was tested

`python3 tests/run_tests.py` runs **119 tests** against the mock Ollama server in
`tests/fake_ollama.py`, with a temporary database and report directory per test.
They cover: connection failure handling, discovery and `/api/show` parsing,
streaming vs non-streaming generation and TTFT, option sanitisation, the
optimization registry and every family's configuration generation, OFAT bounds,
refinement, evaluation determinism and each scoring criterion, consistency,
summary statistics and confidence intervals, aggregation with failed runs,
normalisation and weighting, ranking under each objective, failure penalties,
database CRUD and cascade deletes, a full end-to-end experiment through report
and PDF, cancellation, failure handling, and every HTTP endpoint including the
SSE stream.

`tests/test_integration.py` runs the same paths against a real Ollama; it skips
itself unless one is reachable.

**No test result, chart or report generated against the mock server is a real
measurement of a real model, and the fixture is labelled as such in its source.**

---

## Limitations

- **Sample size.** 5 runs per configuration exposes large differences but not
  small ones. Confidence intervals are reported; where they overlap, treat the
  configurations as indistinguishable.
- **Prompt dependence.** Results apply to the prompt you benchmarked. A setting
  that helps JSON extraction can hurt creative writing.
- **Evaluator scope.** Quality scores are heuristics over visible output. They
  do not verify factual accuracy for open-ended prompts and do not assess
  reasoning. Raw outputs are included in the report so you can disagree.
- **Hardware and load.** Latency and throughput reflect this machine at this
  moment; background activity moves the numbers.
- **Not tested by design.** GPU layer offloading (`num_gpu`), `keep_alive`
  eviction behaviour, `OLLAMA_NUM_PARALLEL` and multi-client throughput are
  reported as Not tested with reasons, because measuring them would require
  changing server-level settings the app deliberately does not touch.
- **GPU metrics** require `nvidia-smi`; without it they are reported as N/A.
- **Quantization comparison** requires a second local variant of the same model;
  it is never simulated.
- The server is single-user and unauthenticated by design.
