"""
Entry point for LoCha.exe (PyInstaller, see LoCha.spec) and an alternative
way to run LoCha from source: python run_locha.py
"""

import os
import sys


def _log_to_file():
    """A windowed exe has no console, so sys.stdout/sys.stderr are None and
    anything that prints (download progress bars, logging) would crash.
    Send it to a log file instead."""
    folder = os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache"),
        "LoCha",
    )
    os.makedirs(folder, exist_ok=True)
    log = open(os.path.join(folder, "locha.log"), "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


if __name__ == "__main__":
    _log_to_file()
    from LoCha_app.app import main
    main()
