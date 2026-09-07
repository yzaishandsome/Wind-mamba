"""Public entry point for the clean chronological conformal workflow.

The implementation lives with the ASOC revision reproduction scripts so the
same code path can also be imported by the table and figure generators.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "reproduction" / "revision" / "run_conformal_clean.py"
    runpy.run_path(str(script), run_name="__main__")
