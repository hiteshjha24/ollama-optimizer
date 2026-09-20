"""Configuration.

Precedence (highest first):
    1. values persisted through the Settings page (data/settings.json)
    2. environment variables / .env file
    3. built-in defaults

Nothing is hard-coded at the call site: every module reads through ``get_settings()``.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"

_LOCK = threading.RLock()
_CACHE: "Settings | None" = None


def _load_dotenv() -> None:
    """Minimal .env loader (no external dependency)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- Ollama -------------------------------------------------------
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_connect_timeout: float = 5.0
    ollama_request_timeout: float = 120.0
    ollama_retries: int = 2
    ollama_keep_alive: str = "5m"

    # --- Server -------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8848

    # --- Benchmarking defaults ---------------------------------------
    default_runs_per_config: int = 5
    default_max_tokens: int = 512
    default_timeout_seconds: float = 120.0
    default_stream: bool = True
    max_concurrency: int = 1  # conservative: local inference is resource bound
    default_strategy: str = "all"
    default_objective: str = "balanced"
    coarse_fine_search: bool = True
    auto_generate_pdf: bool = True

    # --- Evaluation ---------------------------------------------------
    evaluator_mode: str = "heuristic"  # heuristic | heuristic+llm
    evaluator_model: str = ""          # empty -> reuse benchmarked model
    evaluator_timeout: float = 90.0

    # --- Storage / logging -------------------------------------------
    database_path: str = str(DATA_DIR / "experiments.db")
    report_dir: str = str(DATA_DIR / "reports")
    log_level: str = "INFO"
    log_file: str = str(DATA_DIR / "app.log")

    # runtime-only, not persisted
    _persisted_keys: tuple = field(default=(), repr=False, compare=False)

    def to_public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_persisted_keys", None)
        return d


EDITABLE_KEYS = {
    "ollama_url",
    "ollama_request_timeout",
    "ollama_retries",
    "ollama_keep_alive",
    "default_runs_per_config",
    "default_max_tokens",
    "default_timeout_seconds",
    "default_stream",
    "max_concurrency",
    "default_strategy",
    "default_objective",
    "coarse_fine_search",
    "auto_generate_pdf",
    "evaluator_mode",
    "evaluator_model",
    "report_dir",
    "database_path",
    "log_level",
}


def _from_env() -> Settings:
    _load_dotenv()
    return Settings(
        ollama_url=_env_str("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_connect_timeout=_env_float("OLLAMA_CONNECT_TIMEOUT", 5.0),
        ollama_request_timeout=_env_float("OLLAMA_REQUEST_TIMEOUT", 120.0),
        ollama_retries=_env_int("OLLAMA_RETRIES", 2),
        ollama_keep_alive=_env_str("OLLAMA_KEEP_ALIVE", "5m"),
        host=_env_str("APP_HOST", "127.0.0.1"),
        port=_env_int("APP_PORT", 8848),
        default_runs_per_config=_env_int("DEFAULT_RUNS", 5),
        default_max_tokens=_env_int("DEFAULT_MAX_TOKENS", 512),
        default_timeout_seconds=_env_float("DEFAULT_TIMEOUT", 120.0),
        default_stream=_env_bool("DEFAULT_STREAM", True),
        max_concurrency=_env_int("MAX_CONCURRENCY", 1),
        default_strategy=_env_str("DEFAULT_STRATEGY", "all"),
        default_objective=_env_str("DEFAULT_OBJECTIVE", "balanced"),
        coarse_fine_search=_env_bool("COARSE_FINE_SEARCH", True),
        auto_generate_pdf=_env_bool("AUTO_PDF", True),
        evaluator_mode=_env_str("EVALUATOR_MODE", "heuristic"),
        evaluator_model=_env_str("EVALUATOR_MODEL", ""),
        evaluator_timeout=_env_float("EVALUATOR_TIMEOUT", 90.0),
        database_path=_env_str("DATABASE_PATH", str(DATA_DIR / "experiments.db")),
        report_dir=_env_str("REPORT_DIR", str(DATA_DIR / "reports")),
        log_level=_env_str("LOG_LEVEL", "INFO"),
        log_file=_env_str("LOG_FILE", str(DATA_DIR / "app.log")),
    )


def _apply_overrides(base: Settings, overrides: Dict[str, Any]) -> Settings:
    valid = {f.name for f in fields(Settings)}
    for key, value in overrides.items():
        if key not in valid or key not in EDITABLE_KEYS:
            continue
        current = getattr(base, key)
        try:
            if isinstance(current, bool):
                value = bool(value) if not isinstance(value, str) else value.lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int) and not isinstance(current, bool):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            else:
                value = str(value)
        except (TypeError, ValueError):
            continue
        if key == "ollama_url":
            value = str(value).rstrip("/")
        setattr(base, key, value)
    return base


def get_settings(refresh: bool = False) -> Settings:
    global _CACHE
    with _LOCK:
        if _CACHE is None or refresh:
            settings = _from_env()
            if SETTINGS_FILE.exists():
                try:
                    overrides = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                    if isinstance(overrides, dict):
                        settings = _apply_overrides(settings, overrides)
                except (json.JSONDecodeError, OSError):
                    pass
            _CACHE = settings
        return _CACHE


def save_settings(overrides: Dict[str, Any]) -> Settings:
    """Persist user-editable settings and refresh the cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if SETTINGS_FILE.exists():
        try:
            existing = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    for key, value in overrides.items():
        if key in EDITABLE_KEYS:
            existing[key] = value
    SETTINGS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return get_settings(refresh=True)


def ensure_dirs() -> None:
    settings = get_settings()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path(settings.report_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
