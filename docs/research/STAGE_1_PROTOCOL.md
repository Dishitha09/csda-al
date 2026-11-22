# Research Stage 1: ISIC 2019 preparation protocol

Status: Stage 1 dataset preparation only. This protocol accompanies the generated inventory, protected class statistics, frozen split manifest and preparation QA. No SSL pretraining, classifier fitting, acquisition, difficulty estimation or evaluation of a model is authorized by this stage. Subsequent research stages require explicit review and approval.

## Information boundary

The supplied training collection has an image archive, metadata CSV and ground-truth CSV. Stage 1 validates and joins them by persistent image ID. It may read the complete provided label CSV under a **privileged preparation boundary** to check integrity, partition simulator targets and report dataset distributions. That access must not flow into the active learner.

| Resource | Preparation use | Future active-learner access |
|---|---|---|
| Images and normalized grouping metadata | File validation, identity checks and label-independent grouping | Image-only TRAIN loaders and label-free manifests. |
| Complete ground-truth CSV | One-hot/schema checks, exact ID coverage and descriptive inventory | Never supplied directly to a selector, feature cache or SSL loader. |
| TRAIN target mapping | Separate protected simulator/oracle store | Only queried IDs may reveal targets through the annotation oracle. |
| Validation target mapping | Separate protected availability inventory | May support a later declared validation/difficulty protocol, with its cost disclosed once used. |
| Held-out test target mapping | Separate sealed assessment store | Final locked assessment only; no SSL, selection, tuning or difficulty estimation. |
| Complete and split class counts | Privileged statistics and future reporting definitions | Never a selector input or a reason to redraw the split. |

The simulator starts with **zero acquired TRAIN labels and zero charged TRAIN annotations**, even though a protected copy of the original public targets exists. This is a simulated annotation accounting rule, not a claim that the original ISIC labels were created without effort. Human/data-preparation access to privileged statistics must be distinguished from the algorithm's permitted observations.

## Split freeze

The default preparation rule is a seeded, label-independent grouped **70/15/15 TRAIN/validation/test split, seed 42**. The fractions are targets; connected group sizes determine achieved counts. The realized counts and split fingerprint must be recorded rather than asserting exact fractions.

1. Validate unique image IDs and exact image/metadata/truth coverage. Do not silently discard unmatched rows or corrupt images.
2. Normalize grouping missing values. Strip whitespace and recognize the explicit case-insensitive tokens `''`, `nan`, `none`, `null`, `na`, `n/a`, `<na>`, `nat`, `unknown` and `unk` as absent. An absent ID must never become a shared artificial patient/lesion group.
3. Form connected groups using available patient and lesion identities; preserve transitive relationships. Include verified label-free duplicate groups when the reviewed preparation policy supplies them.
4. Generate and freeze assignments **before using class labels to summarize the realized split**. The splitter must not receive the truth mapping. Do not rerun seeds to obtain preferred class proportions or guarantee minority coverage using sealed labels.
5. Check that known patient/lesion connected components do not cross splits and retain source/content/group fingerprints. Changed images, metadata or grouping rules invalidate the previous freeze.

If patient IDs are absent, state that fact. Lesion grouping and image-content checks do not establish patient independence. Do not infer patient IDs from filenames, age, sex or site. Missing lesion IDs remain missing; affected records receive no invented lesion identity.

Exact file/content duplicates and perceptual near-duplicate candidates are different observations. A perceptual-hash threshold produces candidates that require a declared review/grouping policy; it does not prove clinical identity. This statistical module does not select that threshold or automatically merge candidate pairs. The Stage 1 preparation report records the actual duplicate policy and its consequences separately.

## Statistical definitions

`csda_al.data.analysis` computes statistics from validated observations; no class counts are hardcoded.

- `analyze_class_distribution(labels, assignments=None)` accepts an image-ID-to-class mapping or an integer label sequence. Optional split assignments must cover exactly the same IDs or align exactly with the sequence. It returns overall and per-split eight-class supports, proportions, missing classes and totals.
- `analyze_metadata(rows)` reports normalized missingness, observed-value counts, numeric summaries where applicable and duplicate/missing image-ID observations. It does not impute values or infer patient identity.
- `analyze_dataset(records, label_mapping, split_assignments, metadata_raw=None)` requires identical record/label/split ID universes, computes distributions, checks known patient/lesion group overlaps and transitive components, records a reporting-only minority definition and records initial annotation availability.
- `export_analysis(summary, output_dir, dataset_name='ISIC 2019')` writes JSON, CSV tables and standalone PNG/SVG class-count and split-count figures. The caller must choose a privileged statistics destination, outside any selector input directory.

The fixed diagnostic order is `MEL, NV, BCC, AK, BKL, DF, VASC, SCC`. For observed image count `N`, class counts `n_c` and proportions `p_c=n_c/N`:

```text
Imbalance ratio = max_c(n_c) / min_c(n_c)
Shannon entropy in bits = -sum_{c:p_c>0} p_c log2(p_c)
Effective class count = 2 ** Shannon entropy
Normalized entropy = Shannon entropy / log2(8)
```

When any declared class has zero support, the all-eight imbalance ratio is `null`, with missing classes reported. The observed-class-only ratio is reported separately; it must not replace the eight-class statistic silently. Empty observations have undefined entropy/effective class count, serialized as `null`. These are **dataset distribution statistics**, not model accuracy, prediction entropy, NT-Xent loss or acquisition results.

Every table uses counts derived from the actual supplied collection or a clearly named synthetic unit-test fixture. Logarithmic figure axes preserve large class-count differences; zero counts are annotated as zero without inventing positive bars. Figures explicitly state that they are dataset inventory, with no model training or performance results.

## Fixed minority reporting definition

The Stage 1 rule is predeclared as **the three smallest classes by count in the frozen TRAIN partition**. Ties use the fixed eight-class index order. A zero-support class remains eligible for this reporting definition and is separately flagged as missing evidence.

The selected class identities and counts are computed only after the label-independent split is frozen, then saved for future evaluation reporting. They must not change from round to round according to acquired-pool frequencies. They must not be provided to SUPClust, a difficulty selector, a sampler, a stopping rule or a class quota.

This reporting convention does not establish which classes are clinically most important or intrinsically most difficult. It also differs from future dynamic acquired-frequency head/middle/tail diagnostic reports; those definitions must be named separately. No class-aware stopping behavior is introduced.

## Validation availability and annotation economy

Stage 1 records separate quantities:

- The number of protected simulator TRAIN targets available.
- Acquired TRAIN targets: **0**.
- Charged TRAIN annotations: **0**.
- The number of validation labels available in the frozen partition.
- Validation labels used for acquisition/difficulty: **none in Stage 1**.
- The potential cost of using all available validation labels later.
- Test labels available solely for later locked assessment, with no model/selection use now.

Preparation availability is not a declaration that validation supervision is free. Before any later model or strategy uses validation labels for difficulty, tuning or model selection, the protocol must declare their annotation cost. Charge each unique available validation label included in that supervised resource once; do not charge it again each epoch or round. If a later design chooses a smaller validation-label resource, it must predeclare and freeze that resource and account for it explicitly.

The full 15% validation partition can be much larger than a future TRAIN query budget of 8 or 40 labels. Report the actual available count from the generated summary; no exact count is assumed from the target fraction. A future comparison must show both TRAIN-query and total target-supervision costs, give baselines comparable access, and distinguish a zero-acquired-TRAIN start from zero total target labels. Stage 1 does not choose future budgets, classifier settings or SSL training hyperparameters.

## QA gates and limits

Before advancing beyond preparation, review the actual artifact report for exact ID coverage, image decoding/integrity, missing metadata, zero-support classes, achieved split counts, known group overlap, duplicate policy and fingerprints. Any cross-split connected patient/lesion group is a preparation failure to investigate; the analyzer reports it but does not redraw or repair a split itself.

Descriptive full-pool counts cannot establish that a later acquisition strategy is balanced or effective. Missing patient identifiers limit independence claims. Label-independent grouping may leave rare classes poorly represented in validation/test; reporting missingness is preferable to silently using their labels to alter the split. No train/validation/test model metrics, learning curves or experimental checkpoints are generated in Stage 1.

## Executed software verification

The owned test file `tests/test_dataset_analysis.py` checks hand-calculated supports/entropy, uniform and empty inventories, mapping joins, rejected malformed targets/assignments, normalized null values, group transitivity, placeholder exclusion, deterministic minority ties, zero initial annotations, CSV totals and PNG/SVG output.

Actual scoped commands from the workspace root:

```powershell
& '.\.venv-csda\Scripts\python.exe' -m pytest -q csda_al/tests/test_dataset_analysis.py --basetemp '.test-tmp-stage1-analysis'
& '.\.venv-csda\Scripts\python.exe' -m ruff check csda_al/src/csda_al/data/analysis.py csda_al/tests/test_dataset_analysis.py
& '.\.venv-csda\Scripts\python.exe' -m mypy csda_al/src/csda_al/data/analysis.py --ignore-missing-imports --follow-imports=silent --disable-error-code=import-untyped
```

The scoped test run passed **14 tests**, with Matplotlib/Pyparsing dependency deprecation warnings. Ruff reported **All checks passed!** and mypy reported **Success: no issues found in 1 source file**. An initial run hit a Windows permission error in the host TEMP directory before the export test could start; the explicit workspace `--basetemp` run resolved that environmental failure. Generated synthetic class/split count plots were visually inspected for labels, clipping and provenance. Actual ISIC inventories and their plot inspection are recorded in the parent Stage 1 report.

Stage 1 ends after the reviewed data/preparation artifacts are available. Stop for user approval before research Stage 2.
