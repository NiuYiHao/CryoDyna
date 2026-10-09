#!/usr/bin/env python
"""Run development-only pose/structure forward calibration."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryodyna.optpose.calibration import main

if __name__ == "__main__":
    main()
