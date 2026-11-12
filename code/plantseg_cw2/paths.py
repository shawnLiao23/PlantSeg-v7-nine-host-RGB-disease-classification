"""Locate coursework files from source, an installed package, or a notebook."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root() -> Path:
    configured = os.environ.get("CW2_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"CW2_PROJECT_ROOT is not a directory: {root}")
        return root
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "code" / "plantseg_cw2" / "dataset.py").is_file():
                return candidate
    raise FileNotFoundError(
        "Cannot locate the coursework2 project root. Run from the repository "
        "or set CW2_PROJECT_ROOT to its absolute path."
    )


PROJECT_ROOT = find_project_root()
