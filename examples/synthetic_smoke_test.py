"""Run from an installed project: python examples/synthetic_smoke_test.py."""
from csda_al.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run-al", "--config", "examples/sample_config.yaml", "--synthetic", "--dry-run"]))
