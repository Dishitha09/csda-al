"""Runnable architecture commands; dry runs are explicitly synthetic."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from .acquisition import QueryContext, make_strategy
from .config import Config, load_config
from .data.dataset import ImageDataset
from .data.metadata import load_metadata, load_truth
from .data.oracle import LabelOracle
from .data.split import apply_split, grouped_split, save_split
from .data.synthetic import create_synthetic_dataset
from .data.transforms import build_image_transform, build_ssl_transform
from .engine.active_loop import ActiveLearningEngine
from .engine.checkpointing import load_checkpoint, save_checkpoint
from .evaluation.report import EvaluationLock, evaluate_predictions, export_result
from .features.cache import FeatureStore
from .features.extract import extract_features
from .models.classifier import Classifier
from .models.simclr import SimCLR
from .training.ssl_train import train_ssl
from .training.supervised_train import SupervisedTrainer
from .utils.io import atomic_json, read_json, fingerprint
from .utils.seed import seed_everything, seed_worker


def _data(config: Config, *, synthetic: bool, truth_needed: bool = True):
    output = Path(config.experiment.output_dir)
    if synthetic:
        records, truth = create_synthetic_dataset(output / "synthetic_images", config.experiment.seed,
                                                  image_size=config.dataset.image_size)
        split_hash = fingerprint([(r.image_id, r.split, hashlib.sha256(r.image_path.read_bytes()).hexdigest())
                                  for r in records])
        return records, truth, split_hash
    if config.dataset.prepared_dir:
        from .data.prepared import load_bundle_manifest, load_prepared_records
        bundle = config.dataset.prepared_dir
        manifest = load_bundle_manifest(bundle)
        # This helper never opens a prepared truth file. The real AL path uses
        # the oracle factory below; SSL/extraction receives TRAIN images only.
        records = load_prepared_records(bundle, "train")
        return records, {}, manifest["split_fingerprint"]
    records = load_metadata(config.dataset.image_dir, config.dataset.metadata_csv)
    if config.dataset.split_manifest:
        records, manifest = apply_split(records, config.dataset.split_manifest)
    else:
        d = config.dataset
        records, manifest = grouped_split(records, config.experiment.seed,
                                           (d.train_fraction, d.validation_fraction, d.test_fraction))
        save_split(output / "split_manifest.json", manifest)
    # Partition identities are frozen before opening the label CSV.
    truth = load_truth(config.dataset.ground_truth_csv) if truth_needed and config.dataset.ground_truth_csv else {}
    if truth_needed and set(truth) != {r.image_id for r in records}:
        raise ValueError("ISIC truth CSV must cover exactly the frozen image manifest")
    return records, truth, manifest["fingerprint"]


def _loader(records, config: Config, *, ssl: bool = False):
    transform = build_ssl_transform(config.dataset.image_size) if ssl else build_image_transform(config.dataset.image_size)
    dataset = ImageDataset(records, transform)
    batch_size = config.ssl.batch_size if ssl else config.classifier.batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=ssl,
                      drop_last=ssl, num_workers=config.experiment.num_workers,
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config.experiment.seed))


def _representation(config: Config, train_records, split_hash: str, *, synthetic: bool):
    model = SimCLR(projection_dim=config.ssl.projection_dim)
    version = "synthetic_untrained_wiring_only"
    if config.ssl.checkpoint:
        path = Path(config.ssl.checkpoint)
        if not path.is_file():
            raise ValueError(f"SSL checkpoint does not exist: {path}")
        metadata = load_checkpoint(path, model, expected_metadata={"split_fingerprint": split_hash})
        if not synthetic and metadata.get("synthetic", False):
            raise ValueError("A synthetic checkpoint cannot initialize a real ISIC experiment")
        version = hashlib.sha256(path.read_bytes()).hexdigest()
    elif not synthetic:
        raise ValueError("Real feature extraction requires ssl.checkpoint. Run pretrain-ssl first, then configure its path.")
    store = extract_features(model, _loader(train_records, config), device=config.experiment.device,
                             normalization="l2", encoder_version=version,
                             preprocessing_id=f"exif_rgb_resize{config.dataset.image_size}_imagenet_normalization_v2",
                             split_fingerprint=split_hash)
    return model, store


def _configured(args) -> Config:
    config = load_config(args.config) if getattr(args, "config", None) else Config()
    if getattr(args, "output", None):
        config = replace(config, experiment=replace(config.experiment, output_dir=str(Path(args.output).resolve())))
    if getattr(args, "dry_run", False) or getattr(args, "synthetic", False):
        config = replace(config, dataset=replace(config.dataset, image_size=32),
                         ssl=replace(config.ssl, batch_size=4, epochs=1, checkpoint=None),
                         classifier=replace(config.classifier, batch_size=8, epochs=1, pretrained=False),
                         experiment=replace(config.experiment, device="cpu", num_workers=0,
                                            max_annotations=min(config.experiment.max_annotations, 16),
                                            max_rounds=min(config.experiment.max_rounds, 2)))
    seed_everything(config.experiment.seed)
    return config


def prepared_training_inputs(config: Config):
    """Load TRAIN/validation capabilities only; no full truth or test file read."""
    from .data.prepared import load_bundle_manifest, load_prepared_records, load_validation_truth
    if not config.dataset.prepared_dir:
        raise ValueError("Real training requires dataset.prepared_dir from Stage 1 preparation")
    bundle = config.dataset.prepared_dir
    manifest = load_bundle_manifest(bundle)
    train = load_prepared_records(bundle, "train")
    validation = load_prepared_records(bundle, "validation")
    validation_labels = load_validation_truth(bundle)
    oracle = LabelOracle.from_prepared_train(
        bundle, max_annotations=config.experiment.max_annotations,
        ledger_path=Path(config.experiment.output_dir) / "oracle_ledger.json")
    return train, validation, validation_labels, oracle, manifest["split_fingerprint"]


def _run_prepared(config: Config, *, resume: bool) -> dict:
    train, validation, validation_labels, oracle, split_hash = prepared_training_inputs(config)
    ssl_model, features = _representation(config, train, split_hash, synthetic=False)
    classifier = Classifier(pretrained=config.classifier.pretrained, dropout=config.classifier.dropout)
    classifier.initialize_from_ssl(ssl_model)
    trainer = SupervisedTrainer(classifier, device=config.experiment.device, lr=config.classifier.lr,
                                freeze_stage=config.classifier.freeze_stage,
                                weighted_ce=config.classifier.weighted_ce)
    return ActiveLearningEngine(config, train, validation, validation_labels, features, oracle, trainer,
                                synthetic=False).run(resume=resume)


def _dispatch(args) -> dict:
    if args.command == "cold-start":
        if not args.synthetic and not args.features:
            raise ValueError("cold-start without --synthetic requires --features containing a TRAIN FeatureStore")
        if args.synthetic:
            rng = np.random.default_rng(args.seed)
            synthetic_features = np.vstack([rng.normal(size=(8, 16)) + i * 3 for i in range(8)])
            store = FeatureStore(tuple(f"synthetic_{i:03d}" for i in range(64)), synthetic_features,
                                 encoder_version="synthetic_geometry_no_model")
        else:
            store = FeatureStore.load(args.features)
        labeled = tuple(Path(args.labeled_ids).read_text(encoding="utf-8").splitlines()) if args.labeled_ids else ()
        query = make_strategy(args.strategy).select(QueryContext(store, labeled_ids=labeled,
                                                                 batch_budget=args.budget, seed=args.seed))
        result = {"synthetic": args.synthetic, "no_real_training": True, **query.to_dict()}
        if args.output:
            atomic_json(args.output, result)
        return result
    config = _configured(args)
    output = Path(config.experiment.output_dir)
    synthetic = bool(getattr(args, "synthetic", False) or getattr(args, "dry_run", False))
    if args.command == "evaluate":
        if not args.predictions:
            raise ValueError("evaluate requires --predictions NPZ with y_true and probabilities arrays")
        with np.load(args.predictions, allow_pickle=False) as arrays:
            if not {"y_true", "probabilities"}.issubset(arrays.files):
                raise ValueError("Prediction NPZ requires y_true and probabilities")
            lock = EvaluationLock(**read_json(args.lock)) if args.lock else None
            evaluation = evaluate_predictions(arrays["y_true"], arrays["probabilities"], split=args.split, lock=lock,
                         checkpoint_id=args.checkpoint_id, config_fingerprint=config.fingerprint,
                         split_fingerprint=args.split_fingerprint, label_budget=args.label_budget,
                         validation_annotation_cost=args.validation_cost)
        paths = export_result(evaluation, output)
        return {"result": evaluation.to_dict(), "artifacts": {k: str(v) for k, v in paths.items()}}
    if not synthetic and args.command in {"pretrain-ssl", "extract-features", "run-al"}:
        if not config.dataset.prepared_dir:
            raise ValueError("Real training/extraction requires dataset.prepared_dir. Prepare and review Stage 1 first.")
        if args.command == "run-al":
            return _run_prepared(config, resume=args.resume)
    if args.command == "inspect-data" and config.dataset.prepared_dir:
        from .data.prepared import load_bundle_manifest, load_prepared_records
        manifest = load_bundle_manifest(config.dataset.prepared_dir)
        counts = {s: len(load_prepared_records(config.dataset.prepared_dir, s))
                  for s in ("train", "validation", "test")}
        return {"images": sum(counts.values()), "split_counts": counts,
                "split_fingerprint": manifest["split_fingerprint"], "truth_labels_loaded": False,
                "prepared": True, "validation_label_cost_if_used": counts["validation"]}
    records, truth, split_hash = _data(config, synthetic=synthetic, truth_needed=args.command == "run-al")
    train_records = [r for r in records if r.split == "train"]
    if not train_records:
        raise ValueError("Frozen split has no TRAIN images")
    if args.command == "inspect-data":
        return {"images": len(records), "split_counts": {s: sum(r.split == s for r in records)
                for s in ("train", "validation", "test")}, "split_fingerprint": split_hash,
                "truth_labels_loaded": False, "patient_group_warning": not any(r.patient_id for r in records)}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "resolved_config.json", config.to_dict())
    if args.command == "pretrain-ssl":
        model = SimCLR(projection_dim=config.ssl.projection_dim)
        result = train_ssl(model, _loader(train_records, config, ssl=True), device=config.experiment.device,
                           epochs=config.ssl.epochs, max_steps=2 if synthetic else getattr(args, "max_steps", None),
                           temperature=config.ssl.temperature, lr=config.ssl.lr)
        path = output / "simclr.pt"
        save_checkpoint(path, model, metadata={"synthetic": synthetic, "split_fingerprint": split_hash,
                                               "config_fingerprint": config.fingerprint, "steps": result["steps"]})
        return {"synthetic": synthetic, "no_real_training": synthetic, "training": result,
                "checkpoint": str(path)}
    ssl_model, features = _representation(config, train_records, split_hash, synthetic=synthetic)
    features.save(output / "train_features.npz")
    if args.command == "extract-features":
        return {"synthetic": synthetic, "no_real_training": synthetic,
                "feature_shape": list(features.features.shape), "features": str(output / "train_features.npz"),
                "encoder_version": features.encoder_version}
    classifier = Classifier(pretrained=config.classifier.pretrained, dropout=config.classifier.dropout)
    classifier.initialize_from_ssl(ssl_model)
    trainer = SupervisedTrainer(classifier, device=config.experiment.device, lr=config.classifier.lr,
                                freeze_stage=config.classifier.freeze_stage,
                                weighted_ce=config.classifier.weighted_ce)
    validation = [r for r in records if r.split == "validation"]
    val_truth = {r.image_id: truth[r.image_id] for r in validation}
    oracle = LabelOracle({r.image_id: truth[r.image_id] for r in train_records},
                         train_ids=[r.image_id for r in train_records],
                         max_annotations=config.experiment.max_annotations, ledger_path=output / "oracle_ledger.json")
    engine = ActiveLearningEngine(config, train_records, validation, val_truth, features, oracle, trainer,
                                  synthetic=synthetic)
    return engine.run(resume=args.resume, max_train_steps=1 if synthetic else None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSDA-AL architecture; synthetic outputs are not ISIC results")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads (default: 1)")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect-data", "pretrain-ssl", "extract-features", "run-al", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--config", default="configs/experiment.yaml")
        child.add_argument("--output", help="Override run output directory")
        if command != "evaluate":
            child.add_argument("--synthetic", action="store_true", help="Use tiny generated RGB images")
        if command in {"pretrain-ssl", "extract-features", "run-al"}:
            child.add_argument("--dry-run", action="store_true", help="Synthetic CPU execution, at most 2 steps")
            child.add_argument("--max-steps", type=int, default=None, help="Maximum optimizer steps")
        if command == "run-al":
            child.add_argument("--resume", action="store_true")
        if command == "evaluate":
            child.add_argument("--predictions", help="NPZ with y_true and probabilities")
            child.add_argument("--split", choices=("validation", "test"), default="validation")
            child.add_argument("--lock", help="JSON EvaluationLock required for final test")
            child.add_argument("--checkpoint-id", default="")
            child.add_argument("--split-fingerprint", default="")
            child.add_argument("--label-budget", type=int, default=0)
            child.add_argument("--validation-cost", type=int, default=0)
    cold = sub.add_parser("cold-start")
    cold.add_argument("--strategy", default="supclust")
    cold.add_argument("--budget", type=int, default=8)
    cold.add_argument("--seed", type=int, default=42)
    cold.add_argument("--synthetic", action="store_true")
    cold.add_argument("--features", help="Saved TRAIN FeatureStore NPZ")
    cold.add_argument("--labeled-ids", help="Previously acquired IDs, one per line")
    cold.add_argument("--output", help="Optional selected-query JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        result = _dispatch(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"CSDA-AL input/run error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
