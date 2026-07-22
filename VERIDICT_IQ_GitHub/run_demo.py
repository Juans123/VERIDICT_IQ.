"""Punto de entrada local para reproducir la validación técnica sintética."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from veridict_iq.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
