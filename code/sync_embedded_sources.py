"""Keep standalone coursework notebooks' embedded modules in sync with code/."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


CODE = Path(__file__).resolve().parent
MODULES = {
    "evaluate.py": "evaluate.py",
    "validation_overlap_analysis.py": "analysis/validation_overlap_analysis.py",
    "evaluate_extension.py": "analysis/evaluate_extension.py",
    "diagnose_validation_consistency.py": "analysis/diagnose_validation_consistency.py",
}
PREVIOUS_EMBEDDED_SHA256 = {
    "evaluate.py": "2b6fdd2dab02a70e3588808dafd7918baa5d5ca14991d1d2928f7ce64832891c",
    "validation_overlap_analysis.py": "a918cf118b33e45df2ab816a16f642c5f7381a0c9e123194db787c9b27203bf8",
    "evaluate_extension.py": "cb7cd7952ba1410a610d8b0e4fc19aa31e423e9b9d10bfb38b9332a9d67e5512",
}


def embedded_cell(filename: str, relative: str) -> dict:
    source = (CODE / "plantseg_cw2" / relative).read_text(encoding="utf-8")
    if "'''" in source:
        raise ValueError(f"Cannot embed triple single quotes in {relative}")
    text = (
        f"# Embedded source: {filename}\n"
        "import hashlib\n"
        f"module_source = r'''\n{source}'''\n"
        f"module_path = EMBED_DIR / {relative!r}\n"
        "module_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "if module_path.exists():\n"
        "    previous = module_path.read_text(encoding='utf-8')\n"
        "    if previous != module_source.lstrip('\\n'):\n"
        f"        if hashlib.sha256(previous.encode()).hexdigest() == {PREVIOUS_EMBEDDED_SHA256.get(filename)!r}:\n"
        "            module_path.write_text(module_source.lstrip('\\n'), encoding='utf-8')\n"
        "            print('Updated previous embedded source:', module_path)\n"
        "        else:\n"
        "            raise ValueError(f'Existing module differs from embedded source: {module_path}')\n"
        "else:\n"
        "    module_path.write_text(module_source.lstrip('\\n'), encoding='utf-8')\n"
        "print('Ready:', module_path)\n"
    )
    return {"cell_type": "code", "execution_count": None, "id": "diagnose-validation",
            "metadata": {"jupyter": {"source_hidden": True}}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    check = parser.parse_args().check
    for name in ("AIP304_CW2_Option1_Kaggle.ipynb", "AIP304_CW2_Option1.ipynb"):
        path = CODE / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook["cells"]
        for filename, relative in MODULES.items():
            new_cell = embedded_cell(filename, relative)
            indices = [index for index, cell in enumerate(cells)
                       if "".join(cell.get("source", [])).startswith(f"# Embedded source: {filename}\n")]
            if len(indices) > 1:
                raise ValueError(f"Duplicate embedded module {filename} in {path}")
            if indices:
                cell = cells[indices[0]]
                if check and cell["source"] != new_cell["source"]:
                    raise ValueError(f"Embedded source differs: {path} {filename}")
                cell["source"] = new_cell["source"]
            else:
                if check:
                    raise ValueError(f"Embedded source missing: {path} {filename}")
                after = next(index for index, cell in enumerate(cells)
                             if "".join(cell.get("source", [])).startswith(
                                 "# Embedded source: validation_overlap_analysis.py\n"))
                cells.insert(after + 1, new_cell)
        if not check:
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
