from __future__ import annotations

from pathlib import Path


_SHIM_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SHIM_DIR.parent

__path__ = [str(_PROJECT_ROOT)]
