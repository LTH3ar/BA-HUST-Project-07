"""Rerun the analysis end to end by executing the notebooks in order (outputs are written in place).

    python main.py                  # 01 EDA → 02 preprocessing → 03 models → 04 targeting  (≈ 6 min on 2 cores)
    python main.py --skip-training  # keep the committed models in model_output/, rerun 01, 02 and 04
    python main.py --only 04        # one step
    python main.py --dashboard      # after the pipeline: streamlit run dashboard/app.py

Retraining (step 03) can change the models slightly with other library versions, and every downstream number with them.
Use --skip-training to reproduce the numbers quoted in docs/ exactly.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
STEPS = {
    "01": ("notebooks/01_eda.ipynb", "EDA → eda_output/"),
    "02": ("notebooks/02_preprocessing.ipynb", "cleaning, features, splits → data/processed/"),
    "03": ("notebooks/03_modeling.ipynb", "LR / RF / LightGBM, calibration → model_output/"),
    "04": ("notebooks/04_targeting.ipynb", "value, EV targeting, test evaluation, sensitivity → targeting_output/"),
}


def run(step, timeout):
    path, what = STEPS[step]
    print(f"[{step}] {path}: {what}", flush=True)
    t0 = time.perf_counter()
    nb = nbformat.read(ROOT / path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    nbformat.write(nb, ROOT / path)
    print(f"[{step}] done in {time.perf_counter() - t0:,.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-training", action="store_true", help="do not rerun notebook 03; use the committed models")
    ap.add_argument("--only", choices=list(STEPS), help="run a single step")
    ap.add_argument("--dashboard", action="store_true", help="start the Streamlit dashboard afterwards")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds allowed per notebook cell")
    args = ap.parse_args()

    steps = [args.only] if args.only else [s for s in STEPS if not (args.skip_training and s == "03")]
    for s in steps:
        run(s, args.timeout)
    if args.dashboard:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py")], check=False)


if __name__ == "__main__":
    main()
