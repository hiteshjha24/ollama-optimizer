"""Shared test scaffolding: an isolated data directory and settings cache."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, db  # noqa: E402


class SandboxCase(unittest.TestCase):
    """Base case that redirects the database, reports and settings to a temp dir."""

    ollama_url = "http://127.0.0.1:1"   # deliberately dead unless a subclass overrides

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ollama-opt-test-"))
        self._old_settings_file = config.SETTINGS_FILE
        config.SETTINGS_FILE = self.tmp / "settings.json"
        os.environ.update({
            "OLLAMA_URL": self.ollama_url,
            "OLLAMA_RETRIES": "0",
            "OLLAMA_CONNECT_TIMEOUT": "2",
            "OLLAMA_REQUEST_TIMEOUT": "20",
            "DATABASE_PATH": str(self.tmp / "test.db"),
            "REPORT_DIR": str(self.tmp / "reports"),
            "LOG_FILE": str(self.tmp / "app.log"),
            "LOG_LEVEL": "ERROR",
            "DEFAULT_RUNS": "2",
            "AUTO_PDF": "0",
        })
        db.reset_connections()
        config.get_settings(refresh=True)
        (self.tmp / "reports").mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        db.reset_connections()
        config.SETTINGS_FILE = self._old_settings_file
        shutil.rmtree(self.tmp, ignore_errors=True)
        config.get_settings(refresh=True)

    def use_ollama(self, url: str) -> None:
        """Point the whole application at a (fake or real) Ollama instance."""
        os.environ["OLLAMA_URL"] = url
        config.get_settings(refresh=True)
