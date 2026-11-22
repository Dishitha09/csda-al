"""Run actual CPU architecture checks and retain complete command logs."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="Scratch directory outside the deliverable (no recursive cleanup performed)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    logs = root / "docs/test_logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    python = sys.executable
    checks = [
        ("pytest", [python, "-m", "pytest", "-q", "--basetemp", str(work / "pytest"),
                    "--junitxml", str(logs / "pytest.xml")], 0),
        ("ruff", [python, "-m", "ruff", "check", "src", "tests", "examples", "scripts"], 0),
        ("mypy", [python, "-m", "mypy", "src/csda_al"], 0),
        ("inspect_missing_data", [python, "-m", "csda_al.cli", "inspect-data", "--config", "configs/dataset.yaml"], 2),
        ("ssl_dry_run", [python, "-m", "csda_al.cli", "pretrain-ssl", "--config", "configs/simclr.yaml",
                         "--dry-run", "--output", str(work / "ssl")], 0),
        ("feature_dry_run", [python, "-m", "csda_al.cli", "extract-features", "--config", "configs/experiment.yaml",
                             "--dry-run", "--output", str(work / "features")], 0),
        ("cold_start", [python, "-m", "csda_al.cli", "cold-start", "--strategy", "supclust", "--budget", "8", "--synthetic"], 0),
        ("active_loop", [python, "-m", "csda_al.cli", "run-al", "--config", "configs/experiment.yaml",
                         "--synthetic", "--dry-run", "--output", str(work / "al")], 0),
        ("active_loop_resume", [python, "-m", "csda_al.cli", "run-al", "--config", "configs/experiment.yaml",
                                "--synthetic", "--dry-run", "--resume", "--output", str(work / "al")], 0),
        ("evaluate_help", [python, "-m", "csda_al.cli", "evaluate", "--help"], 0),
        ("all_imports", [python, "-c", "import importlib,pkgutil,csda_al; "
                         "mods=list(pkgutil.walk_packages(csda_al.__path__,csda_al.__name__+'.')); "
                         "[importlib.import_module(m.name) for m in mods]; print('Imported',len(mods),'modules')"], 0),
    ]
    records = []
    for name, command, expected in checks:
        started = time.perf_counter()
        result = subprocess.run(command, cwd=root, env=environment, text=True, encoding="utf-8",
                                capture_output=True, timeout=300)
        duration = round(time.perf_counter() - started, 3)
        status = "PASS" if result.returncode == expected else "FAIL"
        if name == "inspect_missing_data" and "image directory does not exist" not in result.stderr:
            status = "FAIL"
        text = ("COMMAND: " + subprocess.list2cmdline(command) + "\nEXPECTED EXIT: " + str(expected)
                + "\nACTUAL EXIT: " + str(result.returncode) + "\nSTATUS: " + status + "\n\nSTDOUT:\n"
                + result.stdout + "\nSTDERR:\n" + result.stderr)
        (logs / f"{name}.txt").write_text(text, encoding="utf-8")
        records.append({"name": name, "status": status, "exit_code": result.returncode,
                        "expected_exit_code": expected, "seconds": duration,
                        "log": f"{name}.txt"})
        print(f"{name}: {status} ({duration}s)", flush=True)
    versions = {}
    for package in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "pandas", "pytest", "ruff", "mypy"):
        versions[package] = importlib.metadata.version(package)
    payload = {"scope": "CPU synthetic architecture verification; NO REAL ISIC TRAINING",
               "python": sys.version, "versions": versions, "checks": records,
               "all_passed": all(r["status"] == "PASS" for r in records)}
    (logs / "verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
