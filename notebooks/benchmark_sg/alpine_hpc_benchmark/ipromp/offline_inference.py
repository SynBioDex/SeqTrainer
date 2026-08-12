#!/usr/bin/env python3
"""Compatibility wrapper for the repository's offline iPro-MP entrypoint."""

from __future__ import annotations

from seqtrainer.adapters.ipromp_inference import main


if __name__ == "__main__":
    raise SystemExit(main())
