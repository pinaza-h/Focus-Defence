"""Entry point for `uv run main.py`."""

import sys

from focus_defence.app import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
