#!/usr/bin/env python
"""Portable CLI for the CryoDyna-optpose benchmark (see docs/optpose.md)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryodyna.optpose.benchmark import main

if __name__ == "__main__":
    main()
