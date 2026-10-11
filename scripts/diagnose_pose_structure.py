#!/usr/bin/env python
"""Run four named oracle development training controls."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryodyna.optpose.disentangle import main

if __name__ == "__main__":
    main()
