"""Strict nested YAML configuration with config-relative paths."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
import yaml

from .utils.io import fingerprint


def positive_int(value: int, name: str, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} integer")


@dataclass(frozen=True)
class DatasetConfig:
    image_dir: str = "../data/ISIC_2019_Training_Input"
    metadata_csv: str | None = None
    ground_truth_csv: str | None = "../data/ISIC_2019_Training_GroundTruth.csv"
    split_manifest: str | None = None
    prepared_dir: str | None = None
    image_size: int = 224
    train_fraction: float = .7
    validation_fraction: float = .15
    test_fraction: float = .15

    def __post_init__(self) -> None:
        positive_int(self.image_size, "image_size")
        f = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(not math.isfinite(x) or x < 0 for x in f) or not math.isclose(sum(f), 1):
            raise ValueError("Dataset split fractions must be finite, nonnegative and sum to one")


@dataclass(frozen=True)
class SSLConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = .001
    temperature: float = .5
    projection_dim: int = 128
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "projection_dim"):
            positive_int(getattr(self, name), name)
        if self.batch_size < 2:
            raise ValueError("SSL batch_size must be at least 2")
        if any(not math.isfinite(v) or v <= 0 for v in (self.lr, self.temperature)):
            raise ValueError("SSL learning rate/temperature must be finite positive")


@dataclass(frozen=True)
class ClassifierConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = .001
    freeze_stage: str = "head"
    pretrained: bool = False
    dropout: float = .1
    weighted_ce: bool = False

    def __post_init__(self) -> None:
        positive_int(self.epochs, "classifier epochs")
        positive_int(self.batch_size, "classifier batch_size")
        if self.freeze_stage not in ("head", "layer4", "all"):
            raise ValueError("freeze_stage must be head, layer4 or all")
        if not math.isfinite(self.lr) or self.lr <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid classifier learning rate or dropout")
        if not isinstance(self.pretrained, bool) or not isinstance(self.weighted_ce, bool):
            raise ValueError("Classifier flags must be boolean")


@dataclass(frozen=True)
class AcquisitionConfig:
    initial_strategy: str = "supclust"
    strategy: str = "difficulty_aware"
    initial_budget: int = 8
    batch_budget: int = 8
    alpha: float = .5
    temperature: float = 1.
    k_neighbors: int = 10
    smoothing: float = 1.
    prior_recall: float = .5
    mc_passes: int = 10
    diversity: bool = False

    def __post_init__(self) -> None:
        for name in ("initial_budget", "batch_budget", "k_neighbors", "mc_passes"):
            positive_int(getattr(self, name), name)
        if not 0 <= self.alpha <= 1 or not 0 <= self.prior_recall <= 1:
            raise ValueError("alpha and prior_recall must be in [0,1]")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Acquisition temperature must be finite positive")
        if not math.isfinite(self.smoothing) or self.smoothing < 0:
            raise ValueError("smoothing must be finite nonnegative")
        if self.mc_passes < 2:
            raise ValueError("BALD needs at least two MC passes")
        if not isinstance(self.diversity, bool):
            raise ValueError("diversity must be boolean")


@dataclass(frozen=True)
class StoppingConfig:
    enabled: bool = False
    min_rounds: int = 3
    patience: int = 2
    min_delta: float = 0.005
    min_per_class: int = 2
    minority_delta: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("stopping.enabled must be boolean")
        positive_int(self.min_rounds, "min_rounds")
        positive_int(self.patience, "patience")
        if self.min_delta < 0 or self.minority_delta < 0 or self.min_per_class < 0:
            raise ValueError("stopping parameters must be nonnegative")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    max_annotations: int = 32
    max_rounds: int = 4
    total_label_cap: int | None = None
    output_dir: str = "../outputs/run"
    device: str = "cpu"
    num_workers: int = 0

    def __post_init__(self) -> None:
        for name in ("seed", "max_annotations", "max_rounds", "num_workers"):
            positive_int(getattr(self, name), name, zero=True)
        if self.total_label_cap is not None:
            positive_int(self.total_label_cap, "total_label_cap", zero=True)
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be cpu or cuda[:index]")


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    schema = {"dataset": DatasetConfig, "ssl": SSLConfig, "classifier": ClassifierConfig,
              "acquisition": AcquisitionConfig, "experiment": ExperimentConfig,
              "stopping": StoppingConfig}
    if not isinstance(raw, dict) or set(raw) - set(schema):
        raise ValueError("Unknown configuration section; expected " + ", ".join(schema))
    parsed = {}
    for section, cls in schema.items():
        values = raw.get(section, {})
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown or invalid configuration fields in {section}")
        values = dict(values)
        # Resolve defaults as well as explicitly provided paths.
        default = cls()
        for key in ("image_dir", "metadata_csv", "ground_truth_csv", "split_manifest", "prepared_dir", "output_dir", "checkpoint"):
            if hasattr(default, key):
                value = values.get(key, getattr(default, key))
                if value is not None:
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"{section}.{key} must be a nonempty path string or null")
                    p = Path(value).expanduser()
                    values[key] = str(p if p.is_absolute() else (path.parent / p).resolve())
        try:
            parsed[section] = cls(**values)
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"Invalid {section} configuration: {exc}") from exc
    return Config(**parsed)
