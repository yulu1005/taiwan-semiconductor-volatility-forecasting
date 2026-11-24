"""Rebuild the supplied V7 outputs from the bundled FinMind snapshot."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent
STEPS = (
    "02_build_dataset.py",
    "03_features_labels.py",
    "03b_features_v2.py",
    "features_30.py",
    "universe.py",
    "03d_garch_oo.py",
    "16_feature_selection_v7.py",
    "05_model1_v7.py",
    "07_model2_v7.py",
    "10_pit_check.py",
    "11_figures_v7.py",
    "14_export_predictions.py",
    "12_correlation_checks.py",
    "15_build_dashboard.py",
    "19_build_research_dashboard.py",
)
RAW_FILES = (
    "price.parquet", "div.parquet", "capred.parquet",
    "listing_dates.csv", "manual_splits.csv", "manifest.json",
)


def prepare():
    raw = ROOT / "data" / "raw"
    for name in RAW_FILES:
        if not (raw / name).is_file():
            raise FileNotFoundError(raw / name)

    run_dir = ROOT / "reproductions" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True)
    shutil.copytree(
        ROOT / "src", run_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(ROOT / "setup.yaml", run_dir)
    shutil.copy2(ROOT / "requirements.txt", run_dir)
    (run_dir / "data" / "raw").mkdir(parents=True)
    for name in RAW_FILES:
        shutil.copy2(raw / name, run_dir / "data" / "raw" / name)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Copy inputs without installing or running.")
    parser.add_argument("--use-current-python", action="store_true", help="Use the current interpreter with dependencies already installed.")
    args = parser.parse_args()

    run_dir = prepare()
    print(f"Run folder: {run_dir}", flush=True)
    if args.prepare_only:
        return

    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    python = sys.executable
    if not args.use_current_python:
        venv_dir = run_dir / ".venv"
        venv.create(venv_dir, with_pip=True)
        python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([python, "-m", "pip", "install", "-r", "requirements.txt"],
                       cwd=run_dir, env=env, check=True)

    for step in STEPS:
        print(f"Running {step}", flush=True)
        log_path = run_dir / f"{Path(step).stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run([python, str(Path("src") / step)], cwd=run_dir, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            sys.exit(f"{step} failed with exit code {result.returncode}. See {log_path}")
    print(f"Research dashboard: {run_dir / 'outputs/v7/dashboard/research_dashboard_preview.html'}")
    print(f"Historical dashboard: {run_dir / 'outputs/v7/dashboard/dashboard.html'}")
    print(f"PIT check: {run_dir / 'outputs/v7/tables/pit/pit_summary.csv'}")


if __name__ == "__main__":
    main()
