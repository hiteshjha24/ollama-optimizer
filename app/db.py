"""SQLite persistence layer.

Entities: Experiment -> Configuration -> Run (metrics + evaluation) and Report.
Uses the standard-library ``sqlite3`` module with one connection per thread and
WAL journaling so the benchmark worker threads and the HTTP threads can share
the database safely.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import get_settings
from .version import SCHEMA_VERSION

_local = threading.local()
_INIT_LOCK = threading.Lock()
_INITIALISED_PATHS: set[str] = set()

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id                TEXT PRIMARY KEY,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    started_at        REAL,
    finished_at       REAL,
    status            TEXT NOT NULL,
    model             TEXT NOT NULL,
    model_meta        TEXT NOT NULL DEFAULT '{}',
    prompt            TEXT NOT NULL,
    prompt_label      TEXT,
    prompt_category   TEXT,
    system_prompt     TEXT,
    strategy          TEXT NOT NULL,
    objective         TEXT NOT NULL,
    weights           TEXT NOT NULL DEFAULT '{}',
    runs_per_config   INTEGER NOT NULL,
    max_tokens        INTEGER NOT NULL,
    timeout_seconds   REAL NOT NULL,
    stream            INTEGER NOT NULL,
    concurrency       INTEGER NOT NULL,
    evaluator_mode    TEXT NOT NULL,
    seed              INTEGER,
    app_version       TEXT,
    ollama_version    TEXT,
    host_info         TEXT NOT NULL DEFAULT '{}',
    progress          TEXT NOT NULL DEFAULT '{}',
    summary           TEXT NOT NULL DEFAULT '{}',
    error             TEXT
);

CREATE TABLE IF NOT EXISTS configurations (
    id             TEXT PRIMARY KEY,
    experiment_id  TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    seq            INTEGER NOT NULL,
    optimization_id TEXT NOT NULL,
    category       TEXT NOT NULL,
    label          TEXT NOT NULL,
    phase          TEXT NOT NULL,
    is_baseline    INTEGER NOT NULL DEFAULT 0,
    options        TEXT NOT NULL DEFAULT '{}',
    prompt         TEXT NOT NULL,
    system_prompt  TEXT,
    rationale      TEXT,
    status         TEXT NOT NULL DEFAULT 'pending',
    skip_reason    TEXT,
    aggregate      TEXT NOT NULL DEFAULT '{}',
    created_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    experiment_id    TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    configuration_id TEXT NOT NULL REFERENCES configurations(id) ON DELETE CASCADE,
    run_index        INTEGER NOT NULL,
    status           TEXT NOT NULL,
    started_at       REAL,
    finished_at      REAL,
    output           TEXT,
    metrics          TEXT NOT NULL DEFAULT '{}',
    resources        TEXT NOT NULL DEFAULT '{}',
    evaluation       TEXT NOT NULL DEFAULT '{}',
    error            TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id            TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    created_at    REAL NOT NULL,
    format        TEXT NOT NULL,
    path          TEXT,
    pages         INTEGER,
    status        TEXT NOT NULL DEFAULT 'ok',
    error         TEXT,
    meta          TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_cfg_exp  ON configurations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_run_exp  ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_run_cfg  ON runs(configuration_id);
CREATE INDEX IF NOT EXISTS idx_rep_exp  ON reports(experiment_id);
CREATE INDEX IF NOT EXISTS idx_exp_time ON experiments(created_at DESC);
"""


def _db_path() -> str:
    return get_settings().database_path


def connect() -> sqlite3.Connection:
    """Return this thread's connection, initialising the schema once per path."""
    path = _db_path()
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "path", None) == path:
        return conn

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")

    with _INIT_LOCK:
        if path not in _INITIALISED_PATHS:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()
            _INITIALISED_PATHS.add(path)

    _local.conn = conn
    _local.path = path
    return conn


def reset_connections() -> None:
    """Drop cached connections (used by tests when the DB path changes)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    _local.conn = None
    _local.path = None


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _j(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: Any, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def create_experiment(data: Dict[str, Any]) -> str:
    conn = connect()
    exp_id = data.get("id") or new_id("exp")
    now = time.time()
    conn.execute(
        """
        INSERT INTO experiments (
            id, created_at, updated_at, status, model, model_meta, prompt,
            prompt_label, prompt_category, system_prompt, strategy, objective, weights,
            runs_per_config, max_tokens, timeout_seconds, stream, concurrency,
            evaluator_mode, seed, app_version, ollama_version, host_info, progress, summary
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            exp_id, now, now, data.get("status", "queued"), data["model"],
            _j(data.get("model_meta", {})), data["prompt"], data.get("prompt_label"),
            data.get("prompt_category"), data.get("system_prompt"),
            data.get("strategy", "all"), data.get("objective", "balanced"),
            _j(data.get("weights", {})), int(data.get("runs_per_config", 5)),
            int(data.get("max_tokens", 512)), float(data.get("timeout_seconds", 120.0)),
            1 if data.get("stream", True) else 0, int(data.get("concurrency", 1)),
            data.get("evaluator_mode", "heuristic"), data.get("seed"),
            data.get("app_version"), data.get("ollama_version"),
            _j(data.get("host_info", {})), _j(data.get("progress", {})),
            _j(data.get("summary", {})),
        ),
    )
    conn.commit()
    return exp_id


def update_experiment(exp_id: str, **fields: Any) -> None:
    if not fields:
        return
    json_fields = {"model_meta", "weights", "host_info", "progress", "summary"}
    sets, values = [], []
    for key, value in fields.items():
        sets.append(f"{key} = ?")
        values.append(_j(value) if key in json_fields else value)
    sets.append("updated_at = ?")
    values.append(time.time())
    values.append(exp_id)
    conn = connect()
    conn.execute(f"UPDATE experiments SET {', '.join(sets)} WHERE id = ?", values)
    conn.commit()


def _experiment_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["model_meta"] = _loads(d.get("model_meta"), {})
    d["weights"] = _loads(d.get("weights"), {})
    d["host_info"] = _loads(d.get("host_info"), {})
    d["progress"] = _loads(d.get("progress"), {})
    d["summary"] = _loads(d.get("summary"), {})
    d["stream"] = bool(d.get("stream"))
    return d


def get_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    row = connect().execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
    return _experiment_row_to_dict(row) if row else None


def list_experiments(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    rows = connect().execute(
        "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [_experiment_row_to_dict(r) for r in rows]


def delete_experiment(exp_id: str) -> bool:
    conn = connect()
    cur = conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
    conn.commit()
    return cur.rowcount > 0


# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------

def create_configuration(experiment_id: str, seq: int, cfg: Dict[str, Any]) -> str:
    conn = connect()
    cfg_id = cfg.get("id") or new_id("cfg")
    conn.execute(
        """
        INSERT INTO configurations (
            id, experiment_id, seq, optimization_id, category, label, phase,
            is_baseline, options, prompt, system_prompt, rationale, status,
            skip_reason, aggregate, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            cfg_id, experiment_id, seq, cfg["optimization_id"], cfg["category"],
            cfg["label"], cfg.get("phase", "coarse"),
            1 if cfg.get("is_baseline") else 0, _j(cfg.get("options", {})),
            cfg["prompt"], cfg.get("system_prompt"), cfg.get("rationale"),
            cfg.get("status", "pending"), cfg.get("skip_reason"),
            _j(cfg.get("aggregate", {})), time.time(),
        ),
    )
    conn.commit()
    return cfg_id


def update_configuration(cfg_id: str, **fields: Any) -> None:
    if not fields:
        return
    json_fields = {"options", "aggregate"}
    sets, values = [], []
    for key, value in fields.items():
        sets.append(f"{key} = ?")
        values.append(_j(value) if key in json_fields else value)
    values.append(cfg_id)
    conn = connect()
    conn.execute(f"UPDATE configurations SET {', '.join(sets)} WHERE id = ?", values)
    conn.commit()


def _cfg_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["options"] = _loads(d.get("options"), {})
    d["aggregate"] = _loads(d.get("aggregate"), {})
    d["is_baseline"] = bool(d.get("is_baseline"))
    return d


def list_configurations(experiment_id: str) -> List[Dict[str, Any]]:
    rows = connect().execute(
        "SELECT * FROM configurations WHERE experiment_id = ? ORDER BY seq ASC",
        (experiment_id,),
    ).fetchall()
    return [_cfg_row_to_dict(r) for r in rows]


def get_configuration(cfg_id: str) -> Optional[Dict[str, Any]]:
    row = connect().execute("SELECT * FROM configurations WHERE id = ?", (cfg_id,)).fetchone()
    return _cfg_row_to_dict(row) if row else None


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

def create_run(experiment_id: str, configuration_id: str, run_index: int,
               status: str = "running", started_at: Optional[float] = None) -> str:
    conn = connect()
    run_id = new_id("run")
    conn.execute(
        """
        INSERT INTO runs (id, experiment_id, configuration_id, run_index, status, started_at)
        VALUES (?,?,?,?,?,?)
        """,
        (run_id, experiment_id, configuration_id, run_index, status,
         started_at if started_at is not None else time.time()),
    )
    conn.commit()
    return run_id


def finish_run(run_id: str, **fields: Any) -> None:
    json_fields = {"metrics", "resources", "evaluation"}
    fields.setdefault("finished_at", time.time())
    sets, values = [], []
    for key, value in fields.items():
        sets.append(f"{key} = ?")
        values.append(_j(value) if key in json_fields else value)
    values.append(run_id)
    conn = connect()
    conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", values)
    conn.commit()


def _run_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["metrics"] = _loads(d.get("metrics"), {})
    d["resources"] = _loads(d.get("resources"), {})
    d["evaluation"] = _loads(d.get("evaluation"), {})
    return d


def list_runs(experiment_id: str, configuration_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = connect()
    if configuration_id:
        rows = conn.execute(
            "SELECT * FROM runs WHERE configuration_id = ? ORDER BY run_index ASC",
            (configuration_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM runs WHERE experiment_id = ? ORDER BY configuration_id, run_index",
            (experiment_id,),
        ).fetchall()
    return [_run_row_to_dict(r) for r in rows]


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def create_report(experiment_id: str, fmt: str, path: Optional[str],
                  pages: Optional[int] = None, status: str = "ok",
                  error: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> str:
    conn = connect()
    rep_id = new_id("rep")
    conn.execute(
        """
        INSERT INTO reports (id, experiment_id, created_at, format, path, pages, status, error, meta)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (rep_id, experiment_id, time.time(), fmt, path, pages, status, error, _j(meta or {})),
    )
    conn.commit()
    return rep_id


def list_reports(experiment_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    conn = connect()
    if experiment_id:
        rows = conn.execute(
            "SELECT * FROM reports WHERE experiment_id = ? ORDER BY created_at DESC",
            (experiment_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM reports ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["meta"] = _loads(d.get("meta"), {})
        out.append(d)
    return out


def latest_report(experiment_id: str, fmt: str) -> Optional[Dict[str, Any]]:
    row = connect().execute(
        "SELECT * FROM reports WHERE experiment_id = ? AND format = ? AND status = 'ok' "
        "ORDER BY created_at DESC LIMIT 1",
        (experiment_id, fmt),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["meta"] = _loads(d.get("meta"), {})
    return d


def counts() -> Dict[str, int]:
    conn = connect()
    def one(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])
    return {
        "experiments": one("SELECT COUNT(*) FROM experiments"),
        "runs": one("SELECT COUNT(*) FROM runs"),
        "reports": one("SELECT COUNT(*) FROM reports"),
        "completed": one("SELECT COUNT(*) FROM experiments WHERE status='completed'"),
        "running": one("SELECT COUNT(*) FROM experiments WHERE status IN ('running','queued')"),
    }
