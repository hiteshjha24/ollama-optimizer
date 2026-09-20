#!/usr/bin/env python3
"""Start the Ollama Optimizer web application.

    python3 run.py                  # http://127.0.0.1:8848
    python3 run.py --port 9000      # different port
    python3 run.py --host 0.0.0.0   # expose on the LAN (see README security note)
    python3 run.py --open           # also open a browser window
"""

from __future__ import annotations

import argparse
import sys

from app.config import ensure_dirs, get_settings
from app.logging_setup import configure_logging
from app.server import serve
from app.version import APP_NAME, APP_VERSION


def main(argv=None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=f"{APP_NAME} {APP_VERSION}")
    parser.add_argument("--host", default=settings.host, help="bind address")
    parser.add_argument("--port", type=int, default=settings.port, help="bind port")
    parser.add_argument("--open", action="store_true", help="open a browser window")
    parser.add_argument("--log-level", default=settings.log_level,
                        help="DEBUG, INFO, WARNING or ERROR")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    ensure_dirs()
    try:
        serve(host=args.host, port=args.port, open_browser=args.open)
    except OSError as exc:
        print(f"Could not start the server on {args.host}:{args.port} - {exc}",
              file=sys.stderr)
        print("Another process may already be using that port; try --port 9000.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
