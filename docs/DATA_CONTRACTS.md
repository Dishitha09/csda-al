# Shared data contracts

> Current implementation status (21 November 2026): Stages B-F were subsequently authorized together and implemented. The Stage A design below records original intent; current APIs and any differences are documented in [MODULE_GUIDE.md](MODULE_GUIDE.md) and the completion report.

Implemented shared contract semantics. Signature sketches describe the information boundary; MODULE_GUIDE.md gives exact current constructors and attributes. All contracts use persistent image IDs rather than row positions. Immutable records and defensive array copies/read-only views prevent accidental mutation after validation. Class order is always `MEL, NV, BCC, AK, BKL, DF, VASC, SCC` (indices 0-7).

## Data ingestion and splits

`ImageRecord(image_id: str, image_path: Path, patient_id: str | None, lesion_id: str | None, split: Split)` is frozen and contains no target. IDs must be nonempty and unique. Unknown grouping fields remain absent; they are never inferred from age/sex or filename heuristics. Paths are resolved against the configured dataset root and checked explicitly. The selector never receives file paths if IDs and features suffice.

The metadata loader validates the official CSV schema and joins by image ID. A separate truth importer accepts exactly one positive entry per row in the fixed eight-class one-hot columns; malformed, missing, ambiguous or unknown-class truth fails. Labels do not enter the image manifest, feature cache or selector logs. Unknown external classes are rejected from the closed-set evaluation contract.

A split manifest is generated before exposing TRAIN labels. Proposed default: seeded label-independent grouped 70/15/15 assignment from the official training collection, with achieved counts recorded rather than assuming exact ratios. Use connected components of shared patient and lesion IDs so transitive relationships stay within one split. If patient IDs are unavailable, use lesion IDs; if neither is available, explicitly flag image-level splitting and the remaining leakage risk. Check duplicate image content and cross-split group collisions; avoid false claims of patient independence. No class-stratified split reads sealed TRAIN truth in the default protocol. Existing validated split manifests can be supplied and frozen.

An independent data-preparation boundary partitions a provided truth CSV into TRAIN-oracle truth, paid validation truth and sealed test truth by those frozen IDs. This preparation can read the supplied labels to construct stores; acquisition and SSL cannot. Hidden-label permutation tests hold splits fixed. File-level separation is an accidental-leakage boundary, not adversarial security against arbitrary Python reflection or filesystem access.

## FeatureStore and prediction alignment

`FeatureStore(image_ids: tuple[str, ...], features: ndarray[N,D], encoder_id: str, encoder_version: str, preprocessing_id: str, normalization: NormalizationMetadata, split_fingerprint: str, schema_version: int)`:

- IDs are unique and rows aligned; dimensions are positive; features are finite float32/float64. Preserve the explicit ID -> row map. Resolve requested IDs or raise, never substitute positional matches.
- Training acquisition feature IDs must equal the frozen TRAIN candidate universe, including acquired members. Validation/test features live in separate stores.
- Normalization metadata names the method and tolerance. For L2 normalization reject zero norms; validate metadata against actual row norms.
- Cache manifest contains shape, dtype, ordered-ID hash, encoder/checkpoint and preprocessing hashes, split hash and payload checksum. NPZ uses `allow_pickle=False`. A cache mismatch invalidates/requires re-extraction; row reordering is explicit.

`PredictionBatch(image_ids, probabilities: ndarray[N,8], class_order, model_checkpoint_id, pool_version)` validates uniqueness, alignment, finiteness, range and row sums. A stochastic prediction batch has shape `[T,N,8]` with `T>=2`, plus dropout configuration/seed. Entropy, BALD and difficulty weighting operate on matching IDs. Missing/extra IDs and stale pool/model identities fail before annotation.

## Oracle, acquired labels and annotation costs

`LabelOracle.annotate(selected_ids: Sequence[str], *, request_id: str) -> AnnotationReceipt` is the only TRAIN truth-revealing operation. The constructor accepts a copied sealed TRAIN-only mapping and an annotation ledger; no strategy object holds this oracle. Python privacy is not a security claim.

The entire request is validated before any charge: IDs unique, present in TRAIN, not previously acquired, within the remaining query budget, request ID fresh or an exact retry. A new request with an already acquired ID fails. Replaying the same request ID and same ordered IDs returns the prior receipt and charges zero additional labels; replay with changed IDs fails. Persisted receipts provide restart consistency. There is no public `labels`, `truth`, `get_label` or label-returning indexing method.

`AnnotationReceipt(request_id, image_ids, acquired_labels, new_unique_cost, cumulative_training_cost, receipt_hash)` is visible only to the controller, acquired-label repository and trainer. Public acquisition logs contain selected IDs/scores but no unrevealed labels. Post-annotation composition is written to a distinct diagnostics artifact after the current query is committed.

`AnnotationBudget(max_training_annotations, validation_annotation_cost, optional_total_label_cap)` validates nonnegative integers. Without a total-label cap, `remaining_training = max_training_annotations - unique_acquired_train`. With a cap, use `min(max_training_annotations - unique_acquired_train, total_label_cap - validation_annotation_cost - unique_acquired_train)`; a cap below already committed costs is invalid. Validation cost equals the number of unique labels made available on the frozen validation subset, paid up front even when the simulated labels arrive in a CSV. It is not charged again each round. A total-label cap, when configured, reserves that validation cost first. Budget curves report both training queries and `training_queries + validation_annotation_cost`. Test assessment labels have their own disclosure and do not fund acquisition. Zero acquired TRAIN labels is not a claim of zero total target labels if validation labels have been purchased.

## PoolState and transaction state machine

`PoolState(train_candidate_ids, labeled_ids, unlabeled_ids, selection_history, query_round, total_unique_annotations, state_version, split_fingerprint)` validates:

- Unique universe; labeled and unlabeled are disjoint and their union equals the TRAIN universe.
- All selected history IDs belong to TRAIN and appear once in successful acquisitions. No validation/test ID is admissible.
- `total_unique_annotations == len(labeled_ids) == ledger.unique_training_cost`; history round order and request IDs agree with committed receipts.
- Nonnegative budgets and monotonic state/round versions. Failed attempts do not advance a successful query count.

Proposed phases: `READY_TO_QUERY -> REQUEST_PREPARED -> ANNOTATED -> TRAINING_PENDING -> MODEL_READY -> READY_TO_QUERY/TERMINATED`. The authoritative committed snapshot joins pool, acquired-label receipt ledger and phase. Temporary output is written beside the destination, flushed and atomically replaced; only a complete manifest becomes current. An interrupted prepared request is replayed or discarded according to whether a receipt was committed. A paid annotation cannot disappear because training later failed. Concurrent state writers are unsupported and rejected by a run lock.

## QueryContext, AcquisitionStrategy and QueryBatch

`QueryContext(feature_store, train_pool_ids, labeled_ids, unlabeled_ids, batch_budget, seed, pool_version, predictions: PredictionBatch | None, difficulty: DifficultyEstimate | None)` is read-only. It has no oracle, truth mapping, targets, validation examples or training dataset. Model inference is performed by an adapter outside the selector so the context cannot indirectly request target-bearing batches.

Context checks partition consistency, training-only IDs, budget type/range and prerequisite availability. The FeatureStore covers the full TRAIN universe `P=L union U`; the PredictionBatch covers exactly the current unlabeled IDs `U`. These arrays are joined by image ID, not assumed to have equal row counts or positional alignment. A zero budget or empty candidate set returns an empty batch without model/clustering work. Negative budgets and booleans masquerading as integers fail. A requested budget above pool size is capped by the engine and the effective value is logged.

`AcquisitionStrategy.select(context: QueryContext) -> QueryBatch` must be deterministic for the same validated inputs, configuration and seed within a recorded numerical environment. Ties use image IDs; randomness uses an explicit local RNG. It cannot mutate context or annotate. It returns at most the effective budget; built-in strategies should fill a feasible batch using their declared fallback. The controller validates outputs independently before calling the oracle.

`QueryBatch(image_ids, scores: tuple[float | None, ...] | None, diagnostics, method, version, config_fingerprint, pool_version)` contains unique previously unlabeled TRAIN IDs, aligned finite scores where defined and JSON-serializable audit details. Mixed fallback scores use `None` and a reason, never NaN/Infinity or a fabricated SUP value. Diagnostics include requested/effective K, occupied/eligible cluster counts, fallback IDs, typicality policy, tie policy, feature version, uncertainty normalization and score settings.

## DifficultyEstimate

`DifficultyEstimate(class_order, values: ndarray[8], validation_support: ndarray[8], raw_recall: ndarray[8], prior_mask: ndarray[8], smoothing_config, validation_split_hash, validation_annotation_cost, model_checkpoint_id)` permits finite difficulty values in `[0,1]`. Support is a nonnegative integer vector. Raw recall for support-zero classes is explicitly undefined (serialized as null, not invalid JSON NaN). Prior-derived values carry a flag. A missing validation set produces no estimate and triggers a declared baseline strategy; it does not create a fictitious measured vector.

## Trainer and model adapters

`ModelAdapter` defines `predict_proba(unlabeled_loader) -> PredictionBatch`, `extract_embeddings(image_loader) -> FeatureStore`, `save_checkpoint(path, metadata)` and controlled reload. `Trainer.fit(model, acquired_training_loader, validation_loader?, config) -> TrainingResult` receives only acquired TRAIN targets. Supervised training may reject an empty labeled pool; the engine acquires the seed batch first. Validation is never an optimizer input. SSL trainers accept image/two-view batches without targets.

Model/checkpoint identities, freeze stage, optimizer groups, completed steps and real measured training losses are returned. Synthetic smoke losses are labeled synthetic and cannot populate experimental result tables. CPU tests use tiny images/batches and at most a couple of optimization steps. Head-only frozen encoders must preserve buffers, while active heads have gradients.

## EvaluationResult and evaluation lock

`EvaluationResult(split, checkpoint_id, label_budget, validation_annotation_cost, class_order, aggregate_metrics, per_class_metrics, support, confusion_matrix, missing_class_flags, compute_cost, config_fingerprint)` validates an eight-by-eight integer confusion matrix and consistent support/total counts. No test result can become a DifficultyEstimate.

`EvaluationLock(config_fingerprint, split_fingerprint, checkpoint_id, strategy_id, predeclared_metrics)` is required for final test evaluation. Validation output is marked for method selection/difficulty; test output is final assessment. This is an API guard plus documented workflow, not protection against a user manually editing Python. Empty inputs yield explicitly undefined aggregate metrics with flags or a declared validation error, never a misleading zero score.
