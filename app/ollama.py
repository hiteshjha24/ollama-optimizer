"""Dedicated Ollama service layer.

Every HTTP call to Ollama in this application goes through :class:`OllamaService`.
Only documented Ollama endpoints and options are used:

    GET  /api/version   GET /api/tags   GET /api/ps
    POST /api/show      POST /api/generate     POST /api/chat

Timings reported by Ollama are in nanoseconds and are converted to seconds here.
Nothing is invented: if Ollama does not return a field, the corresponding metric
is left as ``None`` and surfaced to the user as "not available".
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import get_settings
from .logging_setup import get_logger

log = get_logger("ollama")

NS = 1_000_000_000.0

# Generation options documented by Ollama. Anything not in this set is rejected
# before it reaches the API so we never invent parameters.
SUPPORTED_OPTIONS = {
    "num_keep", "seed", "num_predict", "top_k", "top_p", "min_p", "typical_p",
    "repeat_last_n", "temperature", "repeat_penalty", "presence_penalty",
    "frequency_penalty", "mirostat", "mirostat_tau", "mirostat_eta",
    "penalize_newline", "stop", "numa", "num_ctx", "num_batch", "num_gpu",
    "main_gpu", "use_mmap", "use_mlock", "num_thread",
}

# Options that force the model to be reloaded into memory when changed.
RELOAD_OPTIONS = {"num_ctx", "num_batch", "num_gpu", "main_gpu", "num_thread",
                  "use_mmap", "use_mlock", "numa"}


class OllamaError(RuntimeError):
    """Base class for all structured Ollama failures."""

    kind = "ollama_error"
    user_message = "Ollama request failed."

    def __init__(self, message: str = "", detail: str = ""):
        super().__init__(message or self.user_message)
        self.detail = detail

    def to_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "message": str(self), "detail": self.detail}


class OllamaUnavailable(OllamaError):
    kind = "unavailable"
    user_message = (
        "Cannot reach Ollama. Start it with `ollama serve`, then check the Ollama "
        "URL on the Settings page."
    )


class OllamaTimeout(OllamaError):
    kind = "timeout"
    user_message = "The Ollama request timed out."


class OllamaModelMissing(OllamaError):
    kind = "model_missing"
    user_message = "The selected model is not available in Ollama."


class OllamaBadResponse(OllamaError):
    kind = "bad_response"
    user_message = "Ollama returned a response this application could not parse."


class OllamaUnsupportedOption(OllamaError):
    kind = "unsupported_option"
    user_message = "The model or runtime rejected one of the requested options."


class Cancelled(RuntimeError):
    """Raised when a caller cancels an in-flight generation."""


@dataclass
class GenerationResult:
    """One generation, with only the metrics Ollama actually reported."""

    text: str = ""
    model: str = ""
    done_reason: Optional[str] = None
    # wall-clock, measured by this client
    wall_seconds: float = 0.0
    time_to_first_token: Optional[float] = None
    # reported by Ollama (seconds, converted from ns)
    total_duration: Optional[float] = None
    load_duration: Optional[float] = None
    prompt_eval_duration: Optional[float] = None
    eval_duration: Optional[float] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    stream_chunks: int = 0
    streamed: bool = False
    raw_final: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.output_tokens and self.eval_duration and self.eval_duration > 0:
            return self.output_tokens / self.eval_duration
        return None

    def to_metrics(self) -> Dict[str, Any]:
        """Metric dict persisted per run. ``None`` means 'not reported'."""
        return {
            "wall_seconds": round(self.wall_seconds, 6),
            "time_to_first_token": (round(self.time_to_first_token, 6)
                                    if self.time_to_first_token is not None else None),
            "total_duration": (round(self.total_duration, 6)
                               if self.total_duration is not None else None),
            "load_duration": (round(self.load_duration, 6)
                              if self.load_duration is not None else None),
            "prompt_eval_duration": (round(self.prompt_eval_duration, 6)
                                     if self.prompt_eval_duration is not None else None),
            "eval_duration": (round(self.eval_duration, 6)
                              if self.eval_duration is not None else None),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "tokens_per_second": (round(self.tokens_per_second, 4)
                                  if self.tokens_per_second is not None else None),
            "output_chars": len(self.text),
            "done_reason": self.done_reason,
            "streamed": self.streamed,
            "stream_chunks": self.stream_chunks,
            "truncated": self.done_reason == "length",
        }


def sanitize_options(options: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    """Drop options Ollama does not document. Returns (clean, dropped_keys)."""
    clean, dropped = {}, []
    for key, value in (options or {}).items():
        if key in SUPPORTED_OPTIONS and value is not None:
            clean[key] = value
        else:
            dropped.append(key)
    return clean, dropped


class OllamaService:
    """Thin, well-behaved client for the local Ollama HTTP API."""

    def __init__(self, base_url: Optional[str] = None,
                 connect_timeout: Optional[float] = None,
                 request_timeout: Optional[float] = None,
                 retries: Optional[int] = None):
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.connect_timeout = connect_timeout or settings.ollama_connect_timeout
        self.request_timeout = request_timeout or settings.ollama_request_timeout
        self.retries = settings.ollama_retries if retries is None else retries
        self._version_cache: Optional[str] = None
        self._show_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # low-level transport
    # ------------------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _open(self, path: str, payload: Optional[Dict[str, Any]] = None,
              timeout: Optional[float] = None, method: Optional[str] = None):
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self._url(path), data=data, headers=headers,
            method=method or ("POST" if payload is not None else "GET"),
        )
        return urllib.request.urlopen(req, timeout=timeout or self.request_timeout)

    def _request_json(self, path: str, payload: Optional[Dict[str, Any]] = None,
                      timeout: Optional[float] = None, retries: Optional[int] = None) -> Any:
        attempts = (self.retries if retries is None else retries) + 1
        last: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                with self._open(path, payload, timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                try:
                    return json.loads(body)
                except json.JSONDecodeError as exc:
                    raise OllamaBadResponse(
                        f"Malformed JSON from {path}.", detail=body[:400]
                    ) from exc
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:400]
                except Exception:  # pragma: no cover - defensive
                    pass
                if exc.code == 404:
                    raise OllamaModelMissing(
                        "Ollama returned 404 for this request. The model may have been "
                        "removed, or this Ollama version does not provide that endpoint.",
                        detail=detail,
                    ) from exc
                if exc.code in (400, 422, 500) and "option" in detail.lower():
                    raise OllamaUnsupportedOption(
                        "Ollama rejected one of the requested options.", detail=detail
                    ) from exc
                raise OllamaError(
                    f"Ollama returned HTTP {exc.code} for {path}.", detail=detail
                ) from exc
            except socket.timeout as exc:
                last = OllamaTimeout(f"Timed out calling {path}.")
            except urllib.error.URLError as exc:
                last = OllamaUnavailable(
                    OllamaUnavailable.user_message, detail=f"{exc.reason} ({self.base_url})"
                )
            except (TimeoutError, ConnectionError) as exc:
                last = OllamaUnavailable(
                    OllamaUnavailable.user_message, detail=f"{exc} ({self.base_url})"
                )
            if attempt < attempts - 1:
                time.sleep(0.4 * (attempt + 1))
        raise last or OllamaUnavailable(OllamaUnavailable.user_message)

    # ------------------------------------------------------------------
    # metadata endpoints
    # ------------------------------------------------------------------
    def get_version(self) -> str:
        data = self._request_json("/api/version", timeout=self.connect_timeout)
        version = str((data or {}).get("version", "")) or "unknown"
        self._version_cache = version
        return version

    def check_health(self) -> Dict[str, Any]:
        """Never raises: returns a status dict suitable for the dashboard."""
        started = time.perf_counter()
        try:
            version = self.get_version()
            models = self.list_models()
            return {
                "connected": True,
                "url": self.base_url,
                "version": version,
                "model_count": len(models),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "error": None,
            }
        except OllamaError as exc:
            return {
                "connected": False,
                "url": self.base_url,
                "version": None,
                "model_count": 0,
                "latency_ms": None,
                "error": exc.to_dict(),
            }

    def list_models(self) -> List[Dict[str, Any]]:
        data = self._request_json("/api/tags", timeout=self.connect_timeout * 3)
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise OllamaBadResponse(
                "Ollama's model list was not in the expected format.",
                detail=str(data)[:400],
            )
        models = []
        for raw in data["models"]:
            if not isinstance(raw, dict):
                continue
            details = raw.get("details") or {}
            size = raw.get("size")
            models.append({
                "name": raw.get("name") or raw.get("model") or "unknown",
                "model": raw.get("model") or raw.get("name") or "unknown",
                "digest": (raw.get("digest") or "")[:12],
                "size_bytes": size if isinstance(size, (int, float)) else None,
                "size_human": human_size(size),
                "modified_at": raw.get("modified_at"),
                "family": details.get("family"),
                "families": details.get("families") or [],
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
                "format": details.get("format"),
                "parent_model": details.get("parent_model") or None,
            })
        models.sort(key=lambda m: m["name"].lower())
        return models

    def get_model_info(self, model: str, use_cache: bool = True) -> Dict[str, Any]:
        with self._lock:
            if use_cache and model in self._show_cache:
                return self._show_cache[model]
        data = self._request_json("/api/show", {"model": model},
                                  timeout=self.connect_timeout * 6)
        if not isinstance(data, dict):
            raise OllamaBadResponse("Unexpected /api/show payload.", detail=str(data)[:400])

        details = data.get("details") or {}
        model_info = data.get("model_info") or {}
        context_length = None
        for key, value in model_info.items():
            if key.endswith(".context_length") and isinstance(value, (int, float)):
                context_length = int(value)
                break
        info = {
            "name": model,
            "family": details.get("family"),
            "families": details.get("families") or [],
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "format": details.get("format"),
            "parent_model": details.get("parent_model") or None,
            "context_length": context_length,
            "capabilities": data.get("capabilities") or [],
            "default_parameters": parse_parameters_block(data.get("parameters")),
            "has_template": bool(data.get("template")),
            "template_supports_system": "system" in str(data.get("template") or "").lower(),
            "modelfile_system": extract_modelfile_system(data.get("modelfile")),
            "model_info_keys": sorted(model_info.keys())[:40],
        }
        with self._lock:
            self._show_cache[model] = info
        return info

    def get_running_models(self) -> List[Dict[str, Any]]:
        """/api/ps — loaded models and their VRAM footprint (may be empty)."""
        try:
            data = self._request_json("/api/ps", timeout=self.connect_timeout * 2, retries=0)
        except OllamaError:
            return []
        out = []
        for raw in (data or {}).get("models", []) or []:
            if not isinstance(raw, dict):
                continue
            out.append({
                "name": raw.get("name") or raw.get("model"),
                "size_bytes": raw.get("size"),
                "size_vram_bytes": raw.get("size_vram"),
                "expires_at": raw.get("expires_at"),
            })
        return out

    def capabilities(self, model: str) -> Dict[str, Any]:
        """Detect what can actually be tested for this model/runtime."""
        version = self._version_cache or self.get_version()
        try:
            info = self.get_model_info(model)
        except OllamaError as exc:
            return {
                "version": version,
                "error": exc.to_dict(),
                "supports_system_prompt": False,
                "supports_json_format": False,
                "context_length": None,
                "supports_num_ctx": False,
                "supports_seed": True,
                "declared_capabilities": [],
            }
        declared = [str(c).lower() for c in info.get("capabilities", [])]
        return {
            "version": version,
            "error": None,
            "declared_capabilities": declared,
            "context_length": info.get("context_length"),
            "supports_system_prompt": bool(
                info.get("template_supports_system") or info.get("has_template")
                or "completion" in declared or not declared
            ),
            "supports_json_format": True,  # /api/generate accepts format=json for all models
            "supports_num_ctx": info.get("context_length") is not None,
            "supports_seed": True,
            "supports_tools": "tools" in declared,
            "supports_vision": "vision" in declared,
            "quantization_level": info.get("quantization_level"),
            "parameter_size": info.get("parameter_size"),
            "family": info.get("family"),
            "default_parameters": info.get("default_parameters", {}),
        }

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def generate(self, model: str, prompt: str, *,
                 system: Optional[str] = None,
                 options: Optional[Dict[str, Any]] = None,
                 stream: bool = True,
                 timeout: Optional[float] = None,
                 keep_alive: Optional[str] = None,
                 fmt: Optional[str] = None,
                 cancel_event: Optional[threading.Event] = None,
                 on_token: Optional[Callable[[str], None]] = None) -> GenerationResult:
        """Run one generation against /api/generate.

        With ``stream=True`` the time-to-first-token is measured directly. With
        ``stream=False`` TTFT is left as ``None`` rather than estimated.
        """
        clean_options, dropped = sanitize_options(options or {})
        if dropped:
            log.debug("Dropped unsupported options for %s: %s", model, dropped)

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": bool(stream),
            "options": clean_options,
        }
        if system:
            payload["system"] = system
        if fmt:
            payload["format"] = fmt
        payload["keep_alive"] = keep_alive or get_settings().ollama_keep_alive

        timeout = timeout or self.request_timeout
        result = GenerationResult(model=model, streamed=bool(stream))
        started = time.perf_counter()

        try:
            resp = self._open("/api/generate", payload, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # pragma: no cover
                pass
            if exc.code == 404:
                raise OllamaModelMissing(
                    f"Model '{model}' was not found by Ollama. It may have been removed "
                    "while the experiment was running.", detail=detail) from exc
            raise OllamaError(f"Ollama returned HTTP {exc.code} during generation.",
                              detail=detail) from exc
        except socket.timeout as exc:
            raise OllamaTimeout(f"Generation timed out after {timeout:.0f}s.") from exc
        except urllib.error.URLError as exc:
            raise OllamaUnavailable(OllamaUnavailable.user_message,
                                    detail=str(exc.reason)) from exc

        chunks: List[str] = []
        final: Dict[str, Any] = {}
        try:
            with resp:
                if stream:
                    for raw_line in resp:
                        if cancel_event is not None and cancel_event.is_set():
                            raise Cancelled("Generation cancelled by user.")
                        if time.perf_counter() - started > timeout:
                            raise OllamaTimeout(
                                f"Generation exceeded the {timeout:.0f}s timeout.")
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("error"):
                            raise OllamaError(str(obj["error"]), detail=line[:400])
                        piece = obj.get("response", "")
                        if piece:
                            if result.time_to_first_token is None:
                                result.time_to_first_token = time.perf_counter() - started
                            chunks.append(piece)
                            result.stream_chunks += 1
                            if on_token is not None:
                                try:
                                    on_token(piece)
                                except Exception:  # never let a UI callback break a run
                                    pass
                        if obj.get("done"):
                            final = obj
                else:
                    body = resp.read().decode("utf-8", errors="replace")
                    try:
                        final = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise OllamaBadResponse("Malformed generation response.",
                                                detail=body[:400]) from exc
                    if final.get("error"):
                        raise OllamaError(str(final["error"]), detail=body[:400])
                    chunks.append(final.get("response", "") or "")
        except socket.timeout as exc:
            raise OllamaTimeout(f"Generation timed out after {timeout:.0f}s.") from exc
        except urllib.error.URLError as exc:
            raise OllamaUnavailable("Lost the connection to Ollama mid-generation.",
                                    detail=str(exc.reason)) from exc

        result.wall_seconds = time.perf_counter() - started
        result.text = "".join(chunks)
        result.raw_final = {k: v for k, v in final.items() if k != "context"}
        result.done_reason = final.get("done_reason")
        result.total_duration = _ns(final.get("total_duration"))
        result.load_duration = _ns(final.get("load_duration"))
        result.prompt_eval_duration = _ns(final.get("prompt_eval_duration"))
        result.eval_duration = _ns(final.get("eval_duration"))
        result.prompt_tokens = _int(final.get("prompt_eval_count"))
        result.output_tokens = _int(final.get("eval_count"))
        return result

    def chat(self, model: str, messages: List[Dict[str, str]], *,
             options: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None,
             fmt: Optional[str] = None) -> GenerationResult:
        """Non-streaming /api/chat call (used by the optional LLM evaluator)."""
        clean_options, _ = sanitize_options(options or {})
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": clean_options,
                   "keep_alive": get_settings().ollama_keep_alive}
        if fmt:
            payload["format"] = fmt
        started = time.perf_counter()
        data = self._request_json("/api/chat", payload,
                                  timeout=timeout or self.request_timeout, retries=0)
        if not isinstance(data, dict):
            raise OllamaBadResponse("Unexpected /api/chat payload.")
        result = GenerationResult(model=model, streamed=False)
        result.text = ((data.get("message") or {}).get("content")) or ""
        result.wall_seconds = time.perf_counter() - started
        result.total_duration = _ns(data.get("total_duration"))
        result.eval_duration = _ns(data.get("eval_duration"))
        result.prompt_tokens = _int(data.get("prompt_eval_count"))
        result.output_tokens = _int(data.get("eval_count"))
        result.done_reason = data.get("done_reason")
        result.raw_final = data
        return result


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _ns(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and value >= 0:
        return value / NS
    return None


def _int(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)):
        return int(value)
    return None


def human_size(size: Any) -> Optional[str]:
    if not isinstance(size, (int, float)) or size <= 0:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return None


def parse_parameters_block(raw: Any) -> Dict[str, Any]:
    """Parse the free-form ``parameters`` string returned by /api/show.

    Example input lines: ``temperature 0.7`` / ``stop "<|eot_id|>"``.
    """
    out: Dict[str, Any] = {}
    if not isinstance(raw, str) or not raw.strip():
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts[0], parts[1].strip().strip('"')
        if key not in SUPPORTED_OPTIONS:
            continue
        if key == "stop":
            out.setdefault("stop", []).append(value)
            continue
        try:
            out[key] = int(value) if value.lstrip("-").isdigit() else float(value)
        except ValueError:
            out[key] = value
    return out


def extract_modelfile_system(modelfile: Any) -> Optional[str]:
    """Pull the SYSTEM directive out of a Modelfile, if present."""
    if not isinstance(modelfile, str):
        return None
    lines = modelfile.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.upper().startswith("SYSTEM"):
            continue
        rest = stripped[6:].strip()
        if rest.startswith('"""'):
            head = rest[3:]
            if '"""' in head:  # single-line triple-quoted SYSTEM
                return head.split('"""')[0].strip() or None
            body = [head]
            for follow in lines[idx + 1:]:
                if '"""' in follow:
                    body.append(follow.split('"""')[0])
                    break
                body.append(follow)
            return "\n".join(body).strip() or None
        return rest.strip('"').strip() or None
    return None
