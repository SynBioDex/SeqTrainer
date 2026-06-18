"""CLI wrapper for preparing ai x bio benchmark splits.

Example:
    python benchmarks/prepare_ai_x_bio_splits.py --drive-root /content/drive/MyDrive
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seqtrainer.benchmarks.ai_x_bio import main


if __name__ == "__main__":
    raise SystemExit(main())
