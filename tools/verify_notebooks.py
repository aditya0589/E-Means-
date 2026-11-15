"""Verify that all notebook code cells execute without any runtime errors."""

import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless validation

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
os.chdir(str(root_dir))

def verify_all():
    for i in range(1, 11):
        nb_files = list(Path("notebooks").glob(f"{i:02d}_*.ipynb"))
        assert len(nb_files) == 1, f"Notebook {i} missing"
        nb_path = nb_files[0]
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        print(f"Testing {nb_path.name} ({len(nb['cells'])} cells)...")
        # Shared global environment with project root on sys.path
        global_env = {"__name__": "__main__", "__file__": str(nb_path)}

        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                try:
                    exec(code, global_env)
                except Exception as exc:
                    print(f"FAILED: {nb_path.name} cell #{idx}: {exc}")
                    raise exc

        print(f"  PASSED: {nb_path.name}")

    print("\n========================================================")
    print("ALL 10 NOTEBOOKS EXECUTED FROM TOP TO BOTTOM WITH ZERO ERRORS!")
    print("========================================================")

if __name__ == "__main__":
    verify_all()
