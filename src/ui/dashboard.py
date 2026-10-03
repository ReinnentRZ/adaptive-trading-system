"""
Streamlit Web Dashboard Proxy / Entrypoint.

Directs execution to src/dashboard/app.py for backward and forward compatibility.
"""

from pathlib import Path
import runpy
import sys

_APP_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"

if __name__ == "__main__" or "streamlit" in sys.modules:
    runpy.run_path(str(_APP_PATH), run_name="__main__")
