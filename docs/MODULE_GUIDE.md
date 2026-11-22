# Module and API guide

This guide describes the executable architecture. It is a map to current interfaces rather than a claim of measured model quality. Mathematical derivations, paper deviations and limitations are recorded in the architecture, research traceability, acquisition documentation and source audits.

## Configuration and data

| Module | Interface and behavior |
|---|---|
| `config.py` | `Config` contains frozen dataset, SSL, classifier, acquisition and experiment settings. `load_config(path)` rejects unknown sections/fields and resolves dataset/checkpoint/output paths relative to the YAML file. `config.to_dict()` records resolved settings; `config.fingerprint` identifies them. |
| `data.metadata` | `CLASSES` fixes `MEL, NV, BCC, AK, BKL, DF, VASC, SCC`. Immutable `ImageRecord` contains ID, path, optional patient/lesion IDs and split; it has no label field. `load_metadata` enumerates actual files and rejects ambiguous image IDs. `load_truth` maps named one-hot columns to the fixed order; truth remains at the application/oracle boundary. |
| `data.split` | `grouped_split(records, seed, fractions, check_content=True)` connects shared patient, lesion and exact-content identities without consulting labels. It returns replaced records and a versioned manifest. `save_split` persists it. `apply_split` checks manifest and dataset fingerprints, rejects duplicate IDs, detects changed content/group metadata and rejects cross-split connected groups. Fingerprints are portable across file relocation. |
| `data.dataset` | `ImageDataset(records, transform)` returns `(image, image_id)` with no target. `LabeledImageDataset(records, labels, transform, acquired_only=...)` returns `(image, acquired_label, image_id)` and validates exact ID coverage and split restrictions. Validation labels are restricted to evaluation use. |
| `data.transforms` | `build_image_transform` supplies train/inference transforms. `build_ssl_transform` returns `TwoViewTransform`, which independently augments one raw image twice. Conservative SSL controls are explicit; lesion-preserving assumptions still need future sensitivity analysis. |
| `data.oracle` | `LabelOracle(train_truth, train_ids=..., max_annotations=..., ledger_path=...)` keeps TRAIN truth private. `annotate(selected_ids, request_id=...)` is the only label-revealing method. It validates unique TRAIN queries and budget, persists receipts atomically and supports retrying the same request without a second charge. A stale oracle instance cannot overwrite newly committed receipts. `unique_annotations`, `annotated_ids` and `ledger_path` expose accounting/identities, not hidden targets. |
| `data.synthetic` | `create_synthetic_dataset` writes tiny deterministic RGB fixtures and returns explicit split records plus simulated oracle truth. These fixtures exercise wiring; their labels and scores are not ISIC results. |

Absolute image paths are runtime locations, while image IDs are the join key. Preserve row/ID alignment when integrating any new loader. Dataset content fingerprints cover bytes and known group metadata, so changing source files requires regenerating and reviewing a split.

## Representation, classifier and training

| Module | Interface and behavior |
|---|---|
| `models.backbone` | `ResNet18Backbone(pretrained=False)` emits 512-dimensional pooled embeddings. Optional torchvision weights are a separately configured initialization; synthetic paths disable downloads. |
| `models.simclr` | `SimCLR(backbone=None, projection_dim=128, hidden_dim=512)` shares an encoder across augmented views. `encode(images)` returns backbone features; `forward(images)` returns projected vectors. `nt_xent_loss(z1,z2,temperature)` normalizes projections, masks self-similarity and averages over all `2N` anchors with cross-view positives. It requires `N>=2`. |
| `models.classifier` | `Classifier(num_classes=8, pretrained=False, backbone=None, dropout=0)` emits eight logits. `encode` extracts pooled features; `predict_proba` applies softmax, leaving mode control to its caller. `initialize_from_ssl(module_or_checkpoint)` strictly loads compatible backbone tensors before the classifier head. |
| `training.base` | `ModelAdapter` and `Trainer` protocols describe the model/fit/predict/extract boundary. Concrete training uses `SupervisedTrainer`; a custom research adapter should implement the same behavior and retain split restrictions. |
| `training.ssl_train` | `train_ssl(model, loader, device, epochs, lr, temperature, max_steps, ...)` optimizes contrastive views from TRAIN images with no labels. `SSLTrainer.fit` wraps this function. Synthetic commands bound optimizer steps to one or two batches. |
| `training.finetune` | `configure_trainable_parameters` supports `head`, `layer4` and `all`. `apply_training_modes` keeps frozen stages and their BatchNorm statistics in evaluation mode. `optimizer_parameter_groups` separates head and backbone learning rates. |
| `training.supervised_train` | `SupervisedTrainer(model,device,lr,freeze_stage,weighted_ce,...)` fits cumulative acquired TRAIN examples only. `fit(loader,epochs,max_steps)` returns actual losses/step counts. `set_freeze_stage` rebuilds parameter groups while retaining state for already-trained parameters. `predict(loader)` and `extract(loader)` return `(image_ids, aligned_numpy_matrix)`. Optional weighted CE derives counts from acquired examples only. |

Frozen SSL embeddings with a head-only classifier and a classifier with an unfrozen backbone are separate modes. The engine uses its configured freeze stage for a run; experiments requiring a stage transition can call `set_freeze_stage` explicitly and record the schedule. Tiny labeled sets may omit diagnoses even though the head has eight outputs.

## Feature storage and neighbor search

`features.extract.infer_batches` performs deterministic inference under evaluation/inference mode, returns IDs with every output row and restores caller module modes in a `finally` block. Embedding mode calls `encode`, never the SimCLR projection head. `extract_features` wraps this behavior into a `FeatureStore`, with optional L2 normalization and explicit encoder/preprocessing/split identities.

`features.cache.FeatureStore(image_ids,features,encoder_id,encoder_version,preprocessing_id,normalization,split_fingerprint)` validates unique IDs, floating finite `[N,D]` values and declared normalization. It copies features into a read-only array. `rows(ids)` selects by persistent ID; `.digest` hashes IDs, dtype, shape and values. `save(path)` writes an atomic NPZ; `load(path,expected_encoder_version=...,expected_preprocessing_id=...,expected_split_fingerprint=...)` verifies payload checksums and rejects stale identities. Runtime caches are excluded from the final source archive.

`features.nearest_neighbors.nearest_neighbors(features,k,query=None,exclude_self=True,use_faiss=False)` returns Euclidean distances and row indices. Self-exclusion is identity-based for a self-query. It clips neighbor counts for tiny sets. Optional FAISS uses exact flat L2 search and falls back to scikit-learn when the optional package is absent; its float32 arithmetic can differ from scikit-learn precision.

## Acquisition and class difficulty

`acquisition.base` defines the selection boundary:

- `PredictionBatch` holds unique IDs, an aligned probability matrix, model checkpoint identity, pool version and fixed class order.
- `QueryContext` holds the feature TRAIN universe, acquired and candidate IDs, batch budget, seed, pool version, optional predictions, aggregate difficulty and optional MC predictions. It has no oracle, training truth or validation targets. Candidate IDs partition the feature universe with acquired IDs.
- `QueryBatch` holds selected IDs, optional scores, diagnostics, method/version, configuration fingerprint and pool version. Strategies use `select(context) -> QueryBatch`; the engine independently validates membership, uniqueness, version and budget afterward.

`acquisition.make_strategy(name, **config)` constructs the following strategies. Unsupported names and options fail explicitly.

| Names | Source and behavior |
|---|---|
| `random` | Seeded sampling over candidate IDs. Also provides the initial random baseline. |
| `kmeans`, `kmeans_representative` | Selects representatives near eligible feature-cluster centers. No class labels enter clustering. |
| `typiclust` | Cluster eligibility and inverse-neighbor-distance typicality without SUP boundary ranking. |
| `supclust` | Requests `n_labeled + batch_budget` clusters over all TRAIN features, avoids occupied clusters, retains typical candidates and maximizes boundary SUP scores using softmax-weighted other-center distances. Diagnostics disclose degenerate-clustering fallbacks. |
| `without_typicality`, `supclust_without_typicality` | SUPClust ablation that removes typicality filtering. |
| `without_sup`, `supclust_without_sup` | SUPClust ablation that removes SUP ranking. |
| `entropy` | Predictive entropy normalized by `log(8)`, with deterministic ID tie breaks. |
| `coreset`, `core_set` | Greedy farthest-first embedding coverage relative to acquired IDs; uses a documented deterministic starting policy when none are acquired. |
| `bald` | Positive mutual information `H(mean p) - mean H(p)` from MC predictions. `mc_dropout_probabilities` activates configured stochastic dropout while preserving BatchNorm statistics, caller modes and RNG state. |
| `difficulty_only` | Model-probability-weighted class difficulty without the entropy term. |
| `difficulty_aware`, `entropy_difficulty` | Experimental fused entropy plus predicted-membership difficulty, optionally with diversity reranking. |

`imbalance.class_difficulty.compute_class_difficulty(validation_true,validation_pred,...)` returns `DifficultyEstimate`, or `None` when no validation targets exist. It records support, raw recall, prior-used flags, smoothing configuration, paid validation annotation count, checkpoint identity and split identity. The default smoothed recall is `(TP + lambda*r0)/(support + lambda)`, with `lambda=1`, `r0=0.5`; difficulty is one minus this recall. Unsupported classes are explicitly prior-derived, or rejected in strict mode.

`imbalance.probability_weighting.expected_difficulty` calculates `D(x)=sum_c p(c|x)*d_c`. `normalize_difficulty` supports identity or candidate min-max normalization and reports constant-score behavior. `DifficultyAwareAcquisition` calculates `alpha*entropy +(1-alpha)*normalized_D`; its optional diversity shortlist still uses only scores and embeddings. `imbalance.diagnostics.post_annotation_composition` consumes labels only after annotation; it cannot inform the current query.

Cold-start selection does not compute class difficulty. With no paid validation labels, the engine records and uses an entropy fallback after training rather than inventing a class-difficulty estimate.

## Engine, logging and checkpoints

`engine.pool_state.PoolState` is immutable and ID-based. Its selected history must match acquired IDs exactly, rounds/versions are nonnegative integers and empty query rounds are rejected. `acquire(ids)` produces the next disjoint labeled/unlabeled partition.

`engine.budget.AnnotationBudget` tracks maximum TRAIN annotations, paid validation cost and optional total cap. `remaining(acquired)` respects both caps and rejects inconsistent committed counts.

`ActiveLearningEngine(config,train_records,validation_records,validation_labels,feature_store,oracle,trainer,synthetic=False)` receives no test data. `run(resume=False,max_train_steps=None)` advances a durable transaction through `ready -> prepared -> training_pending -> evaluation_pending -> ready`. Selected IDs are saved before oracle access; stable request IDs make annotation retries idempotent. Atomic snapshots and a run-level OS writer lock protect committed state. Resume checks the run identity, pool, acquired labels, oracle ledger, feature provenance and checkpoint checksum before another query.

Each completed annotation round trains on all acquired examples, saves a checkpoint, records compute/label counts and emits validation reports. Termination is only fixed annotation/round limits or pool exhaustion. The engine does not invoke Co-teaching/VOG or class-aware stopping.

`engine.checkpointing.save_checkpoint(path,model,metadata=...,optimizer=...,scheduler=...)` atomically records versioned model/optimizer/scheduler state, class order and Python/NumPy/PyTorch RNG state. `load_checkpoint` uses `weights_only=True`, checks model type/class order/metadata, and optionally restores RNG. Runtime checkpoints generated by real optimization or synthetic smoke tests are excluded from the final architecture ZIP.

`utils.seed` seeds libraries and loader workers. `utils.io` provides stable JSON fingerprints, atomic JSON writes and the OS writer lock. `utils.logging.JsonlLogger` writes structured events; the `Logger` protocol permits isolated future logger adapters.

## Evaluation and command entry points

`evaluation.classification_metrics` returns fixed-order aggregate/per-class metrics, support and confusion. `evaluate_predictions` defaults to validation; final test evaluation requires matching `EvaluationLock` identities. `export_result` writes canonical JSON and CSV views, with explicit synthetic provenance when requested. `normalized_aulc`, `compare_aulc`, `label_savings` and `plot_learning_curves` operate on actual supplied observations and never extrapolate absent results. Detailed zero-support/cost/curve rules are in `EVALUATION_PROTOCOL.md`.

`python -m csda_al.cli` exposes `inspect-data`, `pretrain-ssl`, `extract-features`, `cold-start`, `run-al`, and `evaluate`. Dry-run paths generate synthetic data, force CPU, disable pretrained downloads and cap training steps. Real feature extraction requires an explicit compatible SSL checkpoint; a synthetic checkpoint is rejected for a real-data run. The CLI's help documents input locations, output overrides, resume and final-test lock arguments.

## Packaging and verification

`scripts/package_project.py` creates `../CSDA_AL_Complete_Architecture.zip` by default with one `csda_al/` root. It requires all primary modules, configurations, tests, documentation, dependency pins/lock and the completion report. Source files, SVG/Mermaid diagrams and text test logs are included; runtime data, image files, feature arrays, weights, reference repositories, caches, virtual environments, build metadata and temporary test directories are excluded.

The generated `csda_al/ARCHIVE_MANIFEST.json` records each other archive member's byte size and SHA256. It does not hash itself; this exception is explicit. Packaging checks ZIP CRC, member/manifest equality, every SHA256, isolated extraction and extracted file hashes, then verifies package/CLI imports resolve to extracted sources. `--run-tests` additionally runs the test suite with extracted `src` first on `PYTHONPATH`; it never relies on importing an editable original checkout. The script replaces a previous deliverable only after verification succeeds.

Contract tests build disposable fake source trees, prove datasets/weights are excluded, detect changed payloads even when ZIP CRC remains valid, and preserve a prior archive when a required file is missing. They do not package the live project during a unit test.
