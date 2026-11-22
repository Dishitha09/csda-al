# CSDA-AL

**Cold-Start and Difficulty-Aware Deep Active Learning for Imbalanced Skin Lesion Classification**

Complete, original modular PyTorch research architecture for **ISIC 2019**. Fixed class order: `MEL, NV, BCC, AK, BKL, DF, VASC, SCC`. Software verification: **131 tests passed**, plus Ruff, configured Mypy and all required CLI checks. These are implementation checks, not ISIC model results.

## Scope

SimCLR ResNet-18 embeddings -> SUPClust cold start -> training-only oracle -> eight-logit ResNet-18 -> paid-validation class difficulty and unlabeled predictions -> uncertainty/difficulty acquisition -> fixed-budget loop. Baselines include random, K-means representative, TypiClust, entropy, Core-Set, BALD and difficulty-only, plus both SUPClust ablations.

No full ISIC training was performed. No model-performance claims or experimental learning curves are supplied. The ZIP excludes images, model weights, temporary synthetic checkpoints and copied reference repositories. Synthetic checkpoints created when running demos are explicitly tagged and have no clinical or research-performance meaning. The synthetic AL demo uses untrained SSL geometry to exercise integration; the separate SSL smoke performs two real optimizer steps on generated images.

The user authorized all remaining stages with **"do all stages now"** on 21 November 2026. That supersedes the earlier per-stage approval pauses. Stage A reports remain as historical evidence; B-F are implemented.

## Installation

Tested on Windows CPU, Python 3.11.15, PyTorch 2.6.0+cpu and torchvision 0.21.0+cpu. `requirements-lock.txt` records the complete 37-distribution tested environment. CUDA is optional for future experiments and not assumed. The torch/torchvision pair follows [official PyTorch installation guidance](https://pytorch.org/get-started/previous-versions/).

From the extracted `csda_al` directory with uv installed:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
.venv/Scripts/Activate.ps1
```

With an existing Python 3.11 virtual environment, `python -m pip install -r requirements-lock.txt` then `python -m pip install --no-deps -e .` is equivalent. On Linux adjust the virtual-environment executable path to `bin/python`. The exact CPU lock is for the tested environment; other platforms may need the compatible top-level `requirements.txt` plus official platform wheels.

## CPU synthetic commands

```powershell
python -m csda_al.cli pretrain-ssl --config configs/simclr.yaml --dry-run
python -m csda_al.cli extract-features --config configs/experiment.yaml --dry-run
python -m csda_al.cli cold-start --strategy supclust --budget 8 --synthetic
python -m csda_al.cli run-al --config configs/experiment.yaml --synthetic --dry-run
python -m csda_al.cli run-al --config configs/experiment.yaml --synthetic --dry-run --resume
python -m csda_al.cli evaluate --help
python -m pytest -q
```

Use `--output <fresh-directory>` to isolate CLI runs. Existing AL runs require `--resume`; changed configuration, feature provenance, annotation ledger or checkpoint contents are rejected. The default synthetic AL run purchases 16 unique TRAIN labels in two rounds plus eight separately charged validation labels, with two supervised optimizer steps. Eight initial labels do not guarantee eight-class coverage. All output metrics are synthetic wiring observations.

If your system temporary directory is inaccessible, run pytest with `--basetemp=<new-workspace-temp-directory>`. Verification here used a workspace temp directory. Test scratch and `outputs/` are excluded from packaging.

## Later real-data workflow

Supply ISIC images and CSVs yourself; the pack contains none. Set `configs/experiment.yaml` dataset paths. Optional metadata provides lesion/patient groups. Run:

```powershell
python -m csda_al.cli inspect-data --config configs/experiment.yaml
python -m csda_al.cli pretrain-ssl --config configs/experiment.yaml
```

Those real-data commands are implemented but were not executed against ISIC in this assignment. Set `ssl.checkpoint` in the same experiment config to the resulting `simclr.pt`, then use `extract-features` and `run-al` with that config. Use the same frozen grouped split across SSL/acquisition/evaluation. A checkpoint trained on a different split or marked synthetic is rejected. Choose future training epochs/budgets deliberately; the supplied defaults are starting configurations, not validated hyperparameters.

The data loader reads ground truth only after freezing split identities. The oracle receives TRAIN truth only; validation truth feeds metrics/difficulty, and the engine has no test-data input. Validation cost is paid up front and reported alongside training-query cost. Missing validation classes use explicitly flagged priors; no-validation runs use entropy. Patient independence is not claimed when patient metadata is absent.

For saved features, `cold-start --features <train_features.npz> --budget 8 --output query.json` selects IDs without labels. `--labeled-ids <ids.txt>` accepts annotation-status IDs for later clustering rounds. For external evaluation, provide an NPZ with `y_true` and `probabilities`; final `--split test` requires an explicit identity lock. See the evaluation protocol.

## Architecture and verification

- [Completion report](ARCHITECTURE_COMPLETION_REPORT.md)
- [Architecture and Mermaid flow](docs/ARCHITECTURE.md)
- [Actual module APIs](docs/MODULE_GUIDE.md), [data contracts](docs/DATA_CONTRACTS.md)
- [Research traceability](docs/RESEARCH_TRACEABILITY.md), [license notes](LICENSE_NOTES.md)
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md), [research limitations](docs/RESEARCH_LIMITATIONS.md)
- [Stages and actual check logs](docs/STAGE_REVIEWS/STAGE_F.md), [machine verification](docs/test_logs/verification.json)
- [Risk register](docs/RISK_REGISTER.md), [implementation decisions](docs/IMPLEMENTATION_DECISIONS.md)

`src/csda_al/` separates data, models, features, training, acquisition, imbalance, engine, evaluation and utilities. `tests/` contains analytic, leakage, real CPU optimizer, persistence-fault, CLI and packaging tests. Co-teaching/VOG label cleaning, class-aware stopping and ninth-class open-set recognition are outside the primary pipeline.

## Reverification and packaging

```powershell
python scripts/verify_project.py --work-dir ../verification_scratch
python scripts/package_project.py --run-tests
```

The builder creates `../CSDA_AL_Complete_Architecture.zip`, includes a SHA-256 member manifest, verifies CRCs, safely extracts and checks member hashes/imports. `--run-tests` also executes the test suite against the extracted source. The package contains no pretrained checkpoints or benchmark results. No project-wide public reuse license is assigned to the research owner's original work; source/dependency attribution is documented separately.
