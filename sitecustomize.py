from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent
_PARENT_DIR = _PROJECT_ROOT.parent

if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))
