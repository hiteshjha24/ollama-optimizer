"""A mock Ollama server used ONLY by the automated test-suite.

IMPORTANT: nothing this module produces is a real model measurement. It exists so
the unit tests can exercise discovery, the benchmark pipeline, aggregation,
reporting and the API without a local model. Reports produced against this
fixture are clearly synthetic: the model is named ``fake-model`` and the server
is started on localhost by the tests themselves. Never present output generated
against this fixture as a benchmark of a real model.

The responses deliberately mimic the shape of the real Ollama API
(``/api/version``, ``/api/tags``, ``/api/show``, ``/api/ps``, ``/api/generate``,
``/api/chat``) including the nanosecond duration fields, so that the parsing and
metric code under test is the same code that runs in production.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

MODELS = [
    {
        "name": "fake-model:8b",
        "model": "fake-model:8b",
        "size": 4_700_000_000,
        "digest": "sha256:abc123def456",
        "modified_at": "2026-01-02T10:00:00Z",
        "details": {
            "family": "fakellama",
            "families": ["fakellama"],
            "format": "gguf",
            "parameter_size": "8B",
            "quantization_level": "Q4_K_M",
        },
    },
    {
        "name": "fake-model:8b-q8",
        "model": "fake-model:8b-q8",
        "size": 8_100_000_000,
        "digest": "sha256:fff999",
        "modified_at": "2026-01-03T10:00:00Z",
        "details": {
            "family": "fakellama",
            "families": ["fakellama"],
            "format": "gguf",
            "parameter_size": "8B",
            "quantization_level": "Q8_0",
        },
    },
]

# A well-formed answer to the built-in "reasoning_trains" benchmark prompt.
GOOD_TEXT = (
    "Step 1: The trains close a 240 km gap.\n"
    "Step 2: Their combined speed is 60 + 40 = 100 km/h.\n"
    "Step 3: 240 / 100 = 2.4 hours after 10:00.\n"
    "Step 4: 2.4 hours is 2 hours 24 minutes.\n"
    "ANSWER: 12:24"
)
SLOPPY_TEXT = (
    "Well, it depends on a few things, but I think they probably meet somewhere "
    "in the early afternoon. Trains are interesting, aren't they? Anyway the "
    "answer is roughly midday-ish."
)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # keep the test output clean
        pass

    # -- helpers -------------------------------------------------------
    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode())

    # -- routing -------------------------------------------------------
    def do_GET(self) -> None:
        cfg = self.server.config  # type: ignore[attr-defined]
        if cfg.get("offline"):
            self.close_connection = True
            return
        if self.path == "/api/version":
            self._json(200, {"version": cfg.get("version", "0.5.13")})
        elif self.path == "/api/tags":
            self._json(200, {"models": cfg.get("models", MODELS)})
        elif self.path == "/api/ps":
            self._json(200, {"models": []})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        cfg = self.server.config  # type: ignore[attr-defined]
        body = self._read()
        if cfg.get("offline"):
            self.close_connection = True
            return
        if self.path == "/api/show":
            self._show(body, cfg)
        elif self.path == "/api/generate":
            self._generate(body, cfg)
        elif self.path == "/api/chat":
            self._chat(body, cfg)
        else:
            self._json(404, {"error": "not found"})

    # -- endpoints -----------------------------------------------------
    def _show(self, body: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        name = body.get("model") or body.get("name")
        entry = next((m for m in cfg.get("models", MODELS) if m["name"] == name), None)
        if entry is None:
            self._json(404, {"error": f"model '{name}' not found"})
            return
        self._json(200, {
            "details": entry["details"],
            "modelfile": 'FROM fake\nSYSTEM """You are a careful assistant."""',
            "parameters": "stop \"<|end|>\"\ntemperature 0.8",
            "template": "{{ .System }} {{ .Prompt }}",
            "model_info": {
                "general.architecture": "fakellama",
                "fakellama.context_length": 8192,
            },
            "capabilities": ["completion"],
        })

    def _behaviour(self, body: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[str, float]:
        """Deterministic synthetic behaviour: low temperature -> better + faster."""
        options = body.get("options") or {}
        temperature = float(options.get("temperature", 0.8))
        text = GOOD_TEXT if temperature <= 0.45 else SLOPPY_TEXT
        delay = float(cfg.get("delay", 0.004)) * (1.0 + temperature)
        return text, delay

    def _generate(self, body: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        name = body.get("model")
        if name not in {m["name"] for m in cfg.get("models", MODELS)}:
            self._json(404, {"error": f"model '{name}' not found"})
            return
        served = cfg.get("_generate_count", 0) + 1
        cfg["_generate_count"] = served
        fail_after = cfg.get("fail_generate_after")
        if cfg.get("fail_generate") or (fail_after is not None and served > fail_after):
            self._json(500, {"error": "synthetic generation failure"})
            return

        text, delay = self._behaviour(body, cfg)
        prompt_tokens = max(1, len(str(body.get("prompt", "")).split()))
        output_tokens = max(1, len(text.split()))
        eval_ns = int((delay + 0.01) * 1e9)
        final_fields = {
            "model": name,
            "created_at": "2026-01-04T10:00:00Z",
            "done": True,
            "done_reason": "stop",
            "total_duration": eval_ns + 5_000_000,
            "load_duration": 3_000_000,
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": 2_000_000,
            "eval_count": output_tokens,
            "eval_duration": eval_ns,
        }

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            words = text.split(" ")
            for i, word in enumerate(words):
                time.sleep(delay / max(1, len(words)))
                piece = word + ("" if i == len(words) - 1 else " ")
                self._chunk(json.dumps({"model": name, "response": piece, "done": False}) + "\n")
            self._chunk(json.dumps({**final_fields, "response": ""}) + "\n")
            self._chunk("")  # terminating chunk
        else:
            time.sleep(delay)
            self._json(200, {**final_fields, "response": text})

    def _chunk(self, payload: str) -> None:
        data = payload.encode()
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _chat(self, body: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        text, delay = self._behaviour(body, cfg)
        time.sleep(delay)
        # The optional LLM judge asks for JSON scores.
        payload = json.dumps({"correctness": 8, "relevance": 8, "instruction_following": 7,
                              "completeness": 7, "coherence": 8})
        self._json(200, {
            "model": body.get("model"),
            "message": {"role": "assistant", "content": payload},
            "done": True,
            "total_duration": 10_000_000,
            "eval_count": 20,
            "eval_duration": 8_000_000,
            "prompt_eval_count": 40,
            "prompt_eval_duration": 2_000_000,
        })


class FakeOllama:
    """Context manager that runs the mock server on an ephemeral localhost port."""

    def __init__(self, **config: Any):
        self.config: Dict[str, Any] = {"delay": 0.004, **config}
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        assert self.httpd is not None
        host, port = self.httpd.server_address[0], self.httpd.server_address[1]
        return f"http://{host}:{port}"

    def start(self) -> "FakeOllama":
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.config = self.config  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

    def __enter__(self) -> "FakeOllama":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()
