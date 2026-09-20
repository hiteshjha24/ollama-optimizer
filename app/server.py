"""HTTP API + static file server.

Implemented on the standard library's ``ThreadingHTTPServer`` so the whole
application runs with no third-party web framework (see README - Dependency
choices).  The server exposes a small JSON REST API, a Server-Sent Events
stream for live experiment progress, and the static single-page frontend.

Long-running experiments never block an HTTP request: ``POST /api/experiments``
hands the work to :data:`app.engine.MANAGER`, which runs it on a background
thread, and the client follows progress over SSE (or by polling the experiment
endpoint if SSE is unavailable).
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import re
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, unquote

from . import db
from .config import EDITABLE_KEYS, get_settings, save_settings, ensure_dirs
from .engine import MANAGER
from .logging_setup import get_logger
from .ollama import OllamaError, OllamaService, OllamaUnavailable
from .optimizations import STRATEGIES, STRATEGY_LABELS, describe_all
from .prompts import CATEGORIES, list_prompts
from .recommend import OBJECTIVES
from .version import APP_NAME, APP_VERSION

log = get_logger("server")

STATIC_DIR = Path(__file__).resolve().parent / "static"

Handler = Callable[..., Tuple[int, Any]]


class ApiError(Exception):
    def __init__(self, status: int, message: str, detail: str = ""):
        super().__init__(message)
        self.status = status
        self.message = message
        self.detail = detail


# --------------------------------------------------------------------------
# route table
# --------------------------------------------------------------------------

ROUTES: List[Tuple[str, re.Pattern, Handler]] = []


def route(method: str, pattern: str):
    def decorator(fn: Handler) -> Handler:
        ROUTES.append((method, re.compile(f"^{pattern}$"), fn))
        return fn
    return decorator


def _service() -> OllamaService:
    return OllamaService()


# ---- meta / discovery ----------------------------------------------------

@route("GET", r"/api/health")
def api_health(req) -> Tuple[int, Any]:
    health = _service().check_health()
    return 200, {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "ollama": health,
        "counts": db.counts(),
        "running_experiments": MANAGER.running_ids(),
    }


@route("GET", r"/api/models")
def api_models(req) -> Tuple[int, Any]:
    service = _service()
    health = service.check_health()
    if not health["connected"]:
        return 200, {"connected": False, "models": [], "error": health.get("error"),
                     "hint": "Start Ollama with `ollama serve`, then reload this page."}
    try:
        models = service.list_models()
    except OllamaError as exc:
        return 200, {"connected": True, "models": [], "error": exc.to_dict()}
    running = []
    try:
        running = [m.get("name") for m in service.get_running_models()]
    except OllamaError:
        running = []
    return 200, {"connected": True, "version": health.get("version"),
                 "models": models, "loaded": running,
                 "error": None if models else {
                     "message": "Ollama is running but has no models installed.",
                     "detail": "Pull one first, for example: ollama pull llama3.1:8b"}}


@route("GET", r"/api/models/([^/]+)/info")
def api_model_info(req, name: str) -> Tuple[int, Any]:
    model = unquote(name)
    try:
        return 200, _service().capabilities(model)
    except OllamaError as exc:
        raise ApiError(502, str(exc), exc.detail)


@route("GET", r"/api/prompts")
def api_prompts(req) -> Tuple[int, Any]:
    return 200, {"prompts": list_prompts(), "categories": CATEGORIES}


@route("GET", r"/api/optimizations")
def api_optimizations(req) -> Tuple[int, Any]:
    return 200, {
        "optimizations": describe_all(),
        "strategies": [{"id": key, "label": STRATEGY_LABELS.get(key, key),
                        "optimizations": ids} for key, ids in STRATEGIES.items()],
        "objectives": [{"id": key, "weights": weights} for key, weights in OBJECTIVES.items()],
    }


@route("GET", r"/api/dashboard")
def api_dashboard(req) -> Tuple[int, Any]:
    service = _service()
    health = service.check_health()
    models: List[Dict[str, Any]] = []
    if health["connected"]:
        try:
            models = service.list_models()
        except OllamaError:
            models = []
    experiments = db.list_experiments(limit=8)
    return 200, {
        "ollama": health,
        "models_available": len(models),
        "counts": db.counts(),
        "recent": [_experiment_summary(e) for e in experiments],
        "running": MANAGER.running_ids(),
    }


# ---- settings ------------------------------------------------------------

@route("GET", r"/api/settings")
def api_get_settings(req) -> Tuple[int, Any]:
    return 200, {"settings": get_settings().to_public_dict(),
                 "editable": sorted(EDITABLE_KEYS)}


@route("PUT", r"/api/settings")
def api_put_settings(req) -> Tuple[int, Any]:
    body = req.json_body()
    unknown = [k for k in body if k not in EDITABLE_KEYS]
    if unknown:
        raise ApiError(400, f"These settings are not editable: {', '.join(sorted(unknown))}")
    settings = save_settings(body)
    db.reset_connections()
    ensure_dirs()
    return 200, {"settings": settings.to_public_dict(), "saved": sorted(body)}


# ---- experiments ---------------------------------------------------------

def _experiment_summary(exp: Dict[str, Any]) -> Dict[str, Any]:
    summary = exp.get("summary") or {}
    return {
        "id": exp["id"],
        "model": exp["model"],
        "status": exp["status"],
        "created_at": exp["created_at"],
        "finished_at": exp.get("finished_at"),
        "strategy": exp.get("strategy"),
        "objective": exp.get("objective"),
        "prompt_label": exp.get("prompt_label"),
        "prompt_preview": (exp.get("prompt") or "")[:120],
        "headline": summary.get("headline"),
        "configurations_tested": summary.get("configurations_tested"),
        "running": MANAGER.is_running(exp["id"]),
        "error": exp.get("error"),
    }


@route("GET", r"/api/experiments")
def api_list_experiments(req) -> Tuple[int, Any]:
    limit = int(req.query_one("limit", "50"))
    offset = int(req.query_one("offset", "0"))
    rows = db.list_experiments(limit=max(1, min(limit, 200)), offset=max(0, offset))
    return 200, {"experiments": [_experiment_summary(r) for r in rows]}


@route("POST", r"/api/experiments")
def api_start_experiment(req) -> Tuple[int, Any]:
    body = req.json_body()
    if not body.get("model"):
        raise ApiError(400, "Select a model before starting an experiment.")
    if not (body.get("prompt") or "").strip() and not body.get("prompt_id"):
        raise ApiError(400, "Enter a prompt or choose one of the benchmark prompts.")
    try:
        exp_id = MANAGER.start(body)
    except OllamaUnavailable as exc:
        raise ApiError(503, str(exc), exc.detail)
    except OllamaError as exc:
        raise ApiError(502, str(exc), exc.detail)
    except ValueError as exc:
        raise ApiError(400, str(exc))
    return 201, {"experiment_id": exp_id, "status": "started"}


def _require_experiment(exp_id: str) -> Dict[str, Any]:
    exp = db.get_experiment(exp_id)
    if not exp:
        raise ApiError(404, f"Experiment {exp_id} was not found.")
    return exp


@route("GET", r"/api/experiments/([^/]+)")
def api_get_experiment(req, exp_id: str) -> Tuple[int, Any]:
    exp = _require_experiment(exp_id)
    configs = db.list_configurations(exp_id)
    reports = db.list_reports(exp_id)
    exp["running"] = MANAGER.is_running(exp_id)
    return 200, {"experiment": exp, "configurations": configs, "reports": reports}


@route("DELETE", r"/api/experiments/([^/]+)")
def api_delete_experiment(req, exp_id: str) -> Tuple[int, Any]:
    _require_experiment(exp_id)
    if MANAGER.is_running(exp_id):
        raise ApiError(409, "Cancel the experiment before deleting it.")
    db.delete_experiment(exp_id)
    return 200, {"deleted": exp_id}


@route("POST", r"/api/experiments/([^/]+)/cancel")
def api_cancel_experiment(req, exp_id: str) -> Tuple[int, Any]:
    _require_experiment(exp_id)
    ok = MANAGER.cancel(exp_id)
    if not ok:
        raise ApiError(409, "That experiment is not running, so there is nothing to cancel.")
    return 200, {"cancelling": exp_id}


@route("GET", r"/api/experiments/([^/]+)/runs")
def api_experiment_runs(req, exp_id: str) -> Tuple[int, Any]:
    _require_experiment(exp_id)
    cfg_id = req.query_one("configuration_id")
    runs = db.list_runs(exp_id, cfg_id)
    return 200, {"runs": runs}


@route("GET", r"/api/experiments/([^/]+)/progress")
def api_experiment_progress(req, exp_id: str) -> Tuple[int, Any]:
    """Polling fallback for clients that cannot use Server-Sent Events."""
    exp = _require_experiment(exp_id)
    return 200, {"status": exp["status"], "progress": exp.get("progress") or {},
                 "summary": exp.get("summary") or {}, "error": exp.get("error"),
                 "running": MANAGER.is_running(exp_id)}


# ---- reports -------------------------------------------------------------

@route("GET", r"/api/reports")
def api_reports(req) -> Tuple[int, Any]:
    rows = db.list_reports(limit=200)
    for row in rows:
        exp = db.get_experiment(row["experiment_id"])
        row["model"] = exp["model"] if exp else None
        row["exists"] = bool(row.get("path") and Path(row["path"]).exists())
    return 200, {"reports": rows}


@route("GET", r"/api/experiments/([^/]+)/report")
def api_report_json(req, exp_id: str) -> Tuple[int, Any]:
    _require_experiment(exp_id)
    from .report import build_report, render_markdown
    report = build_report(exp_id)
    return 200, {"report": report, "markdown": render_markdown(report),
                 "files": db.list_reports(exp_id)}


@route("POST", r"/api/experiments/([^/]+)/report")
def api_report_generate(req, exp_id: str) -> Tuple[int, Any]:
    """(Re)generate the Markdown report and, unless disabled, the PDF."""
    _require_experiment(exp_id)
    from .report import build_report, write_markdown
    report = build_report(exp_id)
    md_path = write_markdown(report)
    db.create_report(exp_id, "markdown", str(md_path),
                     pages=report["meta"]["estimated_pages"])
    result = {"markdown": str(md_path), "pdf": None, "pdf_error": None,
              "pages": report["meta"]["estimated_pages"]}
    if req.json_body().get("pdf", True):
        try:
            from .pdf import generate_pdf
            pdf_path, pages = generate_pdf(report)
            db.create_report(exp_id, "pdf", str(pdf_path), pages=pages)
            result["pdf"] = str(pdf_path)
            result["pdf_pages"] = pages
        except Exception as exc:  # PDF failure must not lose the Markdown report
            log.error("PDF generation failed for %s: %s", exp_id, exc)
            db.create_report(exp_id, "pdf", None, status="failed", error=str(exc))
            result["pdf_error"] = str(exc)
    return 200, result


def _report_file(exp_id: str, fmt: str) -> Path:
    _require_experiment(exp_id)
    row = db.latest_report(exp_id, fmt)
    if not row or not row.get("path"):
        raise ApiError(404, f"No {fmt} report has been generated for this experiment yet.",
                       (row or {}).get("error") or "")
    path = Path(row["path"])
    if not path.exists():
        raise ApiError(404, f"The {fmt} report file is missing from disk ({path}).",
                       "Regenerate it from the experiment page.")
    return path


# ---- static & SPA --------------------------------------------------------

def _guess_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


# --------------------------------------------------------------------------
# request handler
# --------------------------------------------------------------------------

class RequestHandler(BaseHTTPRequestHandler):
    server_version = f"{APP_NAME.replace(' ', '')}/{APP_VERSION}"
    protocol_version = "HTTP/1.1"

    # -- helpers ------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        log.debug("%s - %s", self.address_string(), fmt % args)

    def json_body(self) -> Dict[str, Any]:
        if getattr(self, "_body_cache", None) is not None:
            return self._body_cache
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            self._body_cache = {}
            return self._body_cache
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(400, "The request body is not valid JSON.", str(exc))
        if not isinstance(data, dict):
            raise ApiError(400, "The request body must be a JSON object.")
        self._body_cache = data
        return data

    def query_one(self, key: str, default: Optional[str] = None) -> Optional[str]:
        values = self._query.get(key)
        return values[0] if values else default

    def _send(self, status: int, body: bytes, content_type: str,
              extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def send_error_json(self, status: int, message: str, detail: str = "") -> None:
        self.send_json(status, {"error": {"status": status, "message": message,
                                          "detail": detail}})

    # -- dispatch -----------------------------------------------------
    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        self._body_cache = None
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        self._query = parse_qs(parsed.query)

        try:
            # Streaming endpoints handle their own response lifecycle.
            if method == "GET" and re.fullmatch(r"/api/experiments/([^/]+)/events", path):
                exp_id = path.split("/")[3]
                self._stream_events(exp_id)
                return
            download = re.fullmatch(r"/api/experiments/([^/]+)/report\.(pdf|md)", path)
            if method == "GET" and download:
                self._send_report_file(download.group(1), download.group(2))
                return

            for route_method, pattern, handler in ROUTES:
                match = pattern.match(path)
                if match and route_method == method:
                    status, payload = handler(self, *match.groups())
                    self.send_json(status, payload)
                    return
                if match and route_method != method:
                    continue

            if path.startswith("/api/"):
                self.send_error_json(404, f"Unknown API endpoint: {path}")
                return
            self._serve_static(path)

        except ApiError as exc:
            self.send_error_json(exc.status, exc.message, exc.detail)
        except OllamaError as exc:
            self.send_error_json(502, str(exc), exc.detail)
        except BrokenPipeError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Unhandled error on %s %s: %s\n%s", method, path, exc,
                      traceback.format_exc())
            try:
                self.send_error_json(500, f"Internal server error: {exc}")
            except Exception:
                pass

    # -- static -------------------------------------------------------
    def _serve_static(self, path: str) -> None:
        rel = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error_json(403, "Forbidden path.")
            return
        if not target.exists() or not target.is_file():
            # single-page app fallback
            target = STATIC_DIR / "index.html"
            if not target.exists():
                self.send_error_json(404, "Frontend assets are missing.")
                return
        self._send(200, target.read_bytes(), _guess_type(target))

    def _send_report_file(self, exp_id: str, fmt: str) -> None:
        kind = "pdf" if fmt == "pdf" else "markdown"
        path = _report_file(exp_id, kind)
        data = path.read_bytes()
        ctype = "application/pdf" if kind == "pdf" else "text/markdown; charset=utf-8"
        self._send(200, data, ctype, {
            "Content-Disposition": f'attachment; filename="{path.name}"'})

    # -- server-sent events -------------------------------------------
    def _stream_events(self, exp_id: str) -> None:
        exp = db.get_experiment(exp_id)
        if not exp:
            self.send_error_json(404, f"Experiment {exp_id} was not found.")
            return

        q = MANAGER.subscribe(exp_id)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        def write(event: Dict[str, Any]) -> None:
            payload = json.dumps(event, ensure_ascii=False, default=str)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            snapshot = db.get_experiment(exp_id) or {}
            write({"type": "snapshot", "status": snapshot.get("status"),
                   "progress": snapshot.get("progress") or {},
                   "summary": snapshot.get("summary") or {}})
            last_ping = time.time()
            while True:
                try:
                    event = q.get(timeout=1.0)
                    write(event)
                    if event.get("type") in {"finished", "failed", "cancelled"}:
                        break
                except queue.Empty:
                    if not MANAGER.is_running(exp_id):
                        current = db.get_experiment(exp_id) or {}
                        if current.get("status") not in {"running", "queued"}:
                            write({"type": "snapshot", "status": current.get("status"),
                                   "progress": current.get("progress") or {},
                                   "summary": current.get("summary") or {}})
                            write({"type": "closed"})
                            break
                    if time.time() - last_ping > 15:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        last_ping = time.time()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            MANAGER.unsubscribe(exp_id, q)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def create_server(host: Optional[str] = None, port: Optional[int] = None) -> ThreadingHTTPServer:
    settings = get_settings()
    ensure_dirs()
    db.connect()  # create schema eagerly so failures surface at startup
    address = (host or settings.host, int(port or settings.port))
    httpd = ThreadingHTTPServer(address, RequestHandler)
    httpd.daemon_threads = True
    return httpd


def serve(host: Optional[str] = None, port: Optional[int] = None,
          open_browser: bool = False) -> None:
    httpd = create_server(host, port)
    host_shown, port_shown = httpd.server_address[0], httpd.server_address[1]
    url = f"http://{host_shown}:{port_shown}"
    print(f"{APP_NAME} {APP_VERSION}")
    print(f"  Serving on {url}")
    health = _service().check_health()
    if health["connected"]:
        print(f"  Ollama: connected (version {health.get('version')})")
    else:
        msg = (health.get("error") or {}).get("message", "not reachable")
        print(f"  Ollama: NOT connected - {msg}")
        print("  Start it with `ollama serve`, then reload the page.")
    print("  Press Ctrl+C to stop.")
    if open_browser:
        threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        httpd.shutdown()
        httpd.server_close()


def _open_browser_later(url: str) -> None:
    import webbrowser
    time.sleep(0.8)
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    serve(open_browser=os.environ.get("OPEN_BROWSER") == "1")
