# Stage B: foundation and data interfaces

The user authorized all stages together after Stage A. This stage implements the data/annotation boundary needed to turn paper acquisition ideas into valid unlabeled-pool software. No real training.

Implemented installable Python 3.11 package/configuration, five YAML configurations, named eight-class CSV ingestion, image-only immutable records, distinct acquired/validation datasets, label-independent patient/lesion/content connected-component splits, content-bound split fingerprints, durable annotation oracle, fixed-budget accounting, immutable pool transitions, atomic JSON, reproducible seeds and OS run locks. These files live in config.py, data/, engine/{budget,pool_state}.py and utils/.

The oracle validates a complete request before charging, rejects non-TRAIN or repeated IDs, and replays matching durable request IDs at zero additional cost. Its public annotated_ids property exposes annotation status only. Validation labels have a separately disclosed up-front cost and total-label caps reserve it. Split assignment does not read diagnostic labels. Final split loading detects changed content, grouping, duplicate IDs and cross-split identities.

Actual tests are test_dataset.py, test_foundation_review.py and the label-boundary tests. Full command/output and per-test cases are in ../test_logs/pytest.txt and pytest.xml; all passed in the final suite. Early checks found a Windows case-insensitive extension issue and stale split/ledger checks; regression tests cover their corrections. Foundation source passes Ruff and configured Mypy. No user ISIC images were available or loaded.

Limitations: grouping depends on available metadata; Python privacy prevents accidental leakage rather than adversarial filesystem access. The oracle is a simulated TRAIN annotation source, not a clinical annotation service. Next stages add models and acquisition using these ID-based interfaces.
