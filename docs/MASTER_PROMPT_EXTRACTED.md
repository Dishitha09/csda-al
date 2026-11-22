# ONE-SHOT ANTIGRAVITY MASTER PROMPT
## Complete Research Project: Imbalance-Aware Active Learning + Cold-Start Transfer-Learning Initialization on ISIC 2019

**COPY EVERYTHING BELOW THIS LINE INTO ANTIGRAVITY AGENT MODE.**

---

You are acting simultaneously as a **senior machine-learning research scientist, research software engineer, reproducibility auditor, medical-image data engineer, statistical analyst, scientific writer, and application designer**. Build and execute a COMPLETE, working, academically defensible, self-contained research project from start to finish. Do not stop after generating an architecture, a skeleton, TODO comments, a few scripts, or hypothetical results. Implement, test, run, analyze, document, package, and verify the real project. Work autonomously and systematically. When a dependency fails, investigate and repair it; do not abandon the project. Keep a durable `PROGRESS.md` and `runs/status.json` after every major stage so work can resume if the agent loses context.

### NON-NEGOTIABLE RULES

1. **One dataset only: ISIC 2019 dermoscopic images**, from the 2019 training set with publicly available ground truth; 25,331 images, eight diagnosis labels `MEL, NV, BCC, AK, BKL, DF, VASC, SCC`. Do not use ISIC 2018, HAM10000 separately, CIFAR, MedMNIST, synthetic images, or a different dataset for research experiments. Synthetic toy feature arrays are allowed ONLY for unit tests and must never appear as ISIC results. The official 2019 challenge test set may contain an unknown class; DO NOT accidentally mix it into the eight-class closed-set experiment. Create an internal held-out test split from the labeled ISIC 2019 training release.
2. The research question is: **Which of two true imbalance-aware active-learning query methods—Class-Balanced Active Learning (Bengar et al., WACV 2022) and CLINICAL (Kothawade et al., 2022)—achieves better label efficiency and rare-class performance on ISIC 2019? Can transfer-learning-based cold-start initialization, diversity, difficulty-aware training, and class-aware stopping improve them?**
3. Compare the **two primary papers' acquisition methods fairly**. Treat FlexSampling (Ju et al.) as a long-tail curriculum/training component, SUPClust (Ono et al.) / TypiClust as initialization or diversity methods, and Khanal et al. as **active re-labeling under noisy labels**, NOT equivalent to normal clean-label acquisition. Do not represent a long-tail training sampler or label-cleaning algorithm as a direct replacement for normal pool-based AL without explicitly describing the adaptation.
4. **Your latest problem statement overrides the older status report:** INCLUDE (a) transfer-learning-based cold-start initialization and (b) a class-aware stopping criterion in the MAIN system, but make **no-transfer** and **no-class-aware-stop** ablation configurations. The older PDF says these were out of scope; this is a documented scope revision, not a reason to omit them.
5. **No fabrication.** Never invent a metric, chart, experiment, citation, dataset download, GPU execution, checkpoint, paper match, or claimed reproducibility. Paper-reported results go in a *separate* table, never mixed with numbers computed by our code. Every reported result must trace to a real `results/` artifact with configuration, seed, budget, and checkpoint hash. If real ISIC 2019 images or compute are genuinely unavailable, complete all feasible implementation/docs/tests, clearly mark experiments `BLOCKED`/`NOT EVALUATED`, and package the runnable project. Never fill missing experiment data with made-up values.
6. **No ground-truth leakage in active learning:** The AL strategy may access the image and image ID of an unlabeled pool item, embeddings, model predictions, validation labels solely for global model selection/approved class-difficulty estimates, and labels of already annotated training items. It may NOT read the hidden true class of a pool example before that exact example has been queried; it may NOT use test labels for training, tuning, checkpoint selection, sampling, target sets, loss weighting, stopping, or early stopping. The benchmark oracle is a separate class with access to ground truth but only reveals requested IDs after acquisition. This constraint applies to all repos being ported.
7. **Real label budget:** Count every newly queried training image exactly once. Always report initial labels, new training labels, cumulative train label count, additionally annotated validation labels, and cumulative total annotation cost. Fixed validation and test-set labels cannot silently be treated as free in claims about total human labeling effort; publish separate AL-only and all-annotation accounting. A class-aware targeted seed must be paid for if used.
8. No unreviewed remote code execution, no committing secrets/API keys, no unlicensed data redistribution. Honor research-code licenses and ISIC 2019 CC BY-NC conditions. The final distribution must NOT contain the ISIC image dataset or proprietary model weights unless explicitly permitted; instead include an executable downloader/manifest and reproducibility steps. The final project and report must say **research only; not for clinical diagnosis**.
9. **Do not ask me to choose routine defaults.** Choose sensible, reproducible defaults and document them. Only request input when authentication/license acceptance or inaccessible ISIC data/GPU fundamentally prevents an action; meanwhile complete all other work. Do not stop after asking. Work on the actual workspace's OS (Windows/PowerShell or Linux shell), not hard-coded paths.
10. At the very end, generate one **complete ZIP** called `ISIC2019_Imbalance_AL_COMPLETE.zip` with the entire executable source, configurations, README, setup/run scripts, tests, bibliographic records, literature review, cleaned methodology, actual result files, charts, report, presentation, demo application and audit. Verify the ZIP can be reopened and its contents used. Put the archive in a user-accessible workspace location and give me the absolute path and a clear way to download/export it. If Antigravity supports an attachment/download artifact, expose it. Do not merely say you will zip it.

### CONTEXT / FILES TO INGEST BEFORE CODING

Search the current working directory and subdirectories for these supplied resources. Their names may include `(1)` or `(2)`; search flexibly. A starter bundle may include them in `inputs/`.

- `AL + Class Imbalance + Cold Start on ISIC 2019 Status Report and Master Prompt.pdf` (9-page PRIOR audit; read it, but see the scope revision above).
- `AL imbalance class(2).zip` (three original PDFs: `2204.03161v2`, `2407.05973v3`, `2403.03741v1`).
- `FlexSampling-main(1).zip` (actual Python code, README, splits; verify if it matches an official version and audit correctness).
- `TypiClust-main(1).zip` (low-budget active-learning source, with associated code).
- `simclr-master(1).zip` (original SimCLR reference, TensorFlow; port to consistent PyTorch implementation if necessary, documenting deviation).
- `imbalanced-medical-active-label-cleaning-main(1).zip` (Khanal MICCAI 2024 implementation).
- `WhatsApp Audio 2026-11-21 at 10.30.23 PM.mp4` (5-minute audio-only material): transcribe locally with a suitable speech recognizer IF available; save `docs/audio_transcript.txt`, extract project requirements, and list any ambiguous instructions. If transcription is not possible, report that fact instead of inventing words; this must not block coding.
- Problem-statement slide image if present: the slide says the pipeline uses transfer-learning-based cold start and a class-aware stopping rule; input is ISIC 2019 dermoscopic images, output is diagnosis prediction among 8 classes with an annotation-efficient AL system.

For each ZIP, safely inspect entry names and audit `README`, licenses, dependencies, actual module source, checkpoints, training logs, and results. Do NOT automatically trust README claims like 'official,' 'reproduced,' or 'completed.' Verify code. In `docs/source_audit.md`, map every paper to its concrete implementation, original repository, exact adaptation, and any missing components. Preserve attribution and licenses in `third_party/`.

**Known high-priority problems that MUST be fixed, with automated regression tests:**
- Supplied FlexSampling `flexsampling/core/anchor.py` uses `labels` across a dataset when selecting class anchors. The supplied `flexsampling/core/curriculum.py::query` accepts `all_labels` of the entire unlabeled pool. `flexsampling/core/pipeline.py` constructs `self._all_labels` from the full dataset and passes it to acquisition. This is **invalid if treated as pool-based AL**: create a strict interface using only labeled-set classes, predicted class probabilities, and unlabeled image embeddings. Keep an unmodified third-party reference and implement a new leak-free AL adaptation. Do not claim it is paper-identical.
- Supplied FlexSampling `examples/run_flexsampling.py` has a `FLEX_SELECT` default of `"cherry"` and picks a maximum test score across checkpoints. It also includes hard-coded `F:/...` image paths. **Never use the cherry mode.** Select checkpoints only on validation macro-F1; evaluate test strictly after model selection. Replace hard-coded paths with configuration.
- Supplied Khanal code contains dataset/HDF5 expectations and possibly Neptune logging/auth placeholders; replace external logging with structured local CSV/JSON. Khanal's problem assumes **initial noisy labels**, while clean-label AL queries new examples. Run the noisy-label arm separately with explicit injected training-label corruption and count *relabeling* instead of *new labels*.
- The PRIOR PDF claimed no official FlexSampling repo; a public repository now exists at `https://github.com/PyJulie/FlexSampling` and its code is among the uploads. Verify and update the audit. Do not treat the older PDF as the final implementation status.
- The supplied archives do **not** include the full 25,331 dermoscopy images and cannot by themselves provide experimental scores. Download the real data from an official source or discover an existing local dataset, verify checksums/counts, then run real experiments.

### ORIGINAL PAPERS — REQUIRED, VERIFIED STARTING POINTS

Research the **original papers** (not just blog summaries), verify authors, venue/year, method details, experimental setting, and current repository status; save `docs/references.bib` with accurate BibTeX and `docs/paper_evidence.csv` with *claim, supported method, exact source, page/section/table/figure, year, caveat*. Follow the citation trail for at least 10 relevant original research papers. The first nine to prioritize:

1. **Bengar et al. (2022), _Class-Balanced Active Learning for Image Classification_**, WACV, `https://openaccess.thecvf.com/content/WACV2022/html/Bengar_Class-Balanced_Active_Learning_for_Image_Classification_WACV_2022_paper.html`, original code `https://github.com/Javadzb/Class-Balanced-AL`; core category: class-aware/balanced acquisition and selection under predicted class imbalance.
2. **Kothawade et al. (2022), _CLINICAL: Targeted Active Learning for Imbalanced Medical Image Classification_**, `https://arxiv.org/abs/2210.01520`; core category: rare-class targeting via submodular mutual information and diversity; original experiments include ISIC **2018**, so adapting to ISIC **2019** is OUR experiment, not reproduction of their ISIC 2018 number.
3. **Ju et al. (2022), _Flexible Sampling for Long-tailed Skin Lesion Classification_**, MICCAI, `https://arxiv.org/abs/2204.03161`; code `https://github.com/PyJulie/FlexSampling`; long-tail curriculum, SSL representations, class prototypes/anchors; explain that it is not a conventional oracle-query AL setup.
4. **Ono et al. (2024), _SUPClust: Active Learning at the Boundaries_**, `https://arxiv.org/abs/2403.03741`; cold start, cluster diversity and boundary/typicality scoring.
5. **Hacohen, Dekel & Weinshall (2022), _Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets_**, ICML, `https://proceedings.mlr.press/v162/hacohen22a.html`; TypiClust for low label budgets, repository `https://github.com/avihu111/TypiClust`.
6. **Khanal et al. (2024), _Active Label Refinement for Robust Training of Imbalanced Medical Image Classification Tasks in the Presence of High Label Noise_**, MICCAI, `https://arxiv.org/abs/2407.05973`; noisy-label cleaning with VOG, code `https://github.com/Bidur-Khanal/imbalanced-medical-active-label-cleaning`.
7. **Doyle et al. (2011), _An active learning based classification strategy for the minority class problem: Application to histopathology annotation_**, BMC Bioinformatics, `https://pmc.ncbi.nlm.nih.gov/articles/PMC3284114/`; class balance and annotation-cost tradeoff in medical imaging.
8. **Sener & Savarese (2018), _Active Learning for Convolutional Neural Networks: A Core-Set Approach_**, ICLR, `https://arxiv.org/abs/1708.00489`; diversity/coverage comparator.
9. **Ash et al. (2020), _Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds_ (BADGE)**, ICLR, `https://arxiv.org/abs/1906.03671`; batch query balancing uncertainty and diversity.

Also investigate, cite, and classify **Chen et al. (2020) SimCLR**, `https://proceedings.mlr.press/v119/chen20j.html` (transferable representations/self-supervised pretraining); **Kim & Yoo (2022) _Blending Query Strategy of Active Learning for Imbalanced Data_** (minority-preferential blending); and one genuine original **cost-sensitive active-learning** paper (not merely weighted cross-entropy). Differentiate *annotation cost for acquiring labels*, *class-dependent misclassification cost*, and *class-balanced training loss*; these are not interchangeable. Include original-paper evidence, strengths, weaknesses, formulas/algorithmic descriptions, and assessment of cold-start suitability. Explicitly distinguish **published results**, **our implementation**, and **our measured results**.

### PHASE 0 — CREATE A COMPLETE, RESUMABLE RESEARCH REPOSITORY

Make a professional, production-quality Python/PyTorch repository; suggested structure (adjust while retaining all capabilities):

```
ISIC2019_Imbalance_AL/
  README.md
  PROGRESS.md
  LICENSES_AND_ATTRIBUTION.md
  CITATION.cff
  pyproject.toml / requirements.lock
  .gitignore  .env.example
  setup.ps1 setup.sh run_all.ps1 run_all.sh
  configs/{full.yaml,smoke.yaml,ablations.yaml,methods/*.yaml}
  inputs/README.md
  papers/                              # supplied research PDFs (license permitting)
  third_party/                         # attributed snapshots/repo links, no secrets
  src/isic_al/{config,data,oracle,model,ssl,initialization,acquisition,
               training,stopping,metrics,experiments,analysis,reporting,utils}/
  scripts/{prepare_data.py,download_isic.py,pretrain.py,run_experiments.py,
           analyze.py,build_report.py,build_slides.py,package.py,verify_archive.py}
  tests/{test_oracle.py,test_splits.py,test_leakage.py,test_acquisition.py,
         test_metrics.py,test_cold_start.py,test_stopping.py,test_resume.py,
         test_experiment_integrity.py,test_bundle.py}
  docs/{literature_review.md,literature_review.pdf,methodology.md,
        source_audit.md,paper_evidence.csv,references.bib,audio_transcript.txt,
        limitations.md,runbook.md,architecture.md}
  data/{raw/,processed/,splits/,manifests/} # raw images ignored/not zipped
  checkpoints/                        # optional; huge model weights excluded from ZIP by default
  runs/{status.json,metadata.json,seeds/...}
  results/{metrics.csv,per_class.csv,queries.csv,budgets.csv,summary.csv,
           tables/,plots/,analysis/,predictions/,final_report.md,final_report.pdf}
  presentation/{ISIC2019_Active_Learning_Research.pptx,figures/}
  dashboard/                           # polished Streamlit research dashboard
  outputs/ISIC2019_Imbalance_AL_COMPLETE.zip
```

Use typed interfaces, docstrings, explicit error messages, structured logs and unit tests; don't add placeholder scripts. Support Windows path spaces, cross-platform installation, CUDA/MPS/CPU detection, checkpoint resume, AMP when appropriate, fixed random seeds, separate config validation, dependency-lock/version metadata, SHA-256 file manifest, and a `--dry-run` that validates environment/data without training. Implement the key logic in actual Python files, not in a notebook-only demonstration.

### PHASE 1 — ISIC 2019 ACQUISITION AND DATA INTEGRITY

1. Search current disk/workspace for official `ISIC_2019_Training_Input` images, `ISIC_2019_Training_GroundTruth.csv`, and training metadata CSV. If absent, download through the **official ISIC site** `https://challenge.isic-archive.com/data/` or official ISIC Archive API. Require true images (not dummy substitute). If connection, storage, or interactive download is blocked, document exact blocker and create a verified downloader and `data/README.md` with commands; continue all other tasks.
2. Validate **25,331 usable unique image IDs** (or log any official discrepancy), one-hot labels across exactly eight known classes, `UNK` never used as a ninth training class, complete image-path mappings, missing/corrupt images, metadata, SHA-256 of CSV/files (or robust manifest). Print actual class counts and imbalance ratio. Audit duplicate images, same lesion IDs and possible same patient identities. Respect anonymization and data licenses.
3. Split **70% train pool / 10% validation / 20% held-out test** with fixed split seed 2026. Prefer **group-aware stratification by lesion ID**, and by patient ID if genuinely available and reliable; groups must not cross splits. Report when perfect class stratification and grouping conflict, class distribution drift, and fallback rules. If IDs missing, use image-level stratification with an explicit leakage warning and duplicate-hash checks. Save `data/splits/split_seed2026.json` and exact ID lists. The held-out test is inaccessible to experiment code until model/checkpoint selection is finalized. Make splits reproducible for all methods/seeds.
4. Apply identical 224x224 preprocessing across methods, color normalization from **train only** when trained from scratch, ImageNet normalization with ImageNet-pretrained backbone when appropriate; train-only augmentations. Validation and test deterministic. No patient leakage between dataset partitions.
5. Build an ISIC image dataloader, robust `get_image(id)`, group-aware indexes, and optionally HDF5 bridge for legacy Khanal code. HDF5 generation must preserve image/label alignment. Legacy `dic.npy` and `train_100.npy` files in FlexSampling are **not** the primary splits unless verified to use exactly the same permitted IDs and no cross-split group overlap.
6. Build a **sealed benchmark-label oracle**: hidden `truth_map` accessible only inside `Oracle.query(ids)`, never directly from acquisition. Before querying, candidates are represented by `UnlabeledRecord(id, image, embedding, prediction)`, with no `label` field. On reveal, append `QueryEvent(method, seed, round, id, revealed_label, cost)`. Assert `queried_ids` have no duplicates and always belong to train pool. Log all class histograms only AFTER queries are revealed.
7. State carefully that holdout validation/test ground truths exist for scientific benchmarking, but those additional annotations represent real costs under a deployable scenario. The train-label query budget, validation annotation count, and full research labeling footprint must be presented separately. For the class-aware stopping rule use **only validation labels** and account for them.

### PHASE 2 — TRANSFER-LEARNING AND COLD-START INITIALIZATION

Main backbone **ResNet-18, 224x224, eight output logits**, initialized from legitimately available **ImageNet-pretrained weights** (transfer learning). Train the classifier head first as the low-label default; define a consistent fine-tuning/unfreezing schedule across arms. If weights cannot be downloaded, provide clear fallback to randomly initialized ResNet-18 but label the result `no_transfer`, not transfer learning. Record exact weight checkpoint/version and SHA-256.

Also implement **SimCLR** pretraining *only on the unlabeled training pool* as a separate self-supervised condition; transfer from external ImageNet and in-domain SSL are different experiments. An `ImageNet + train-only SimCLR` variant may be evaluated as an optional ablation only when computationally justified. `simclr-master` is a TensorFlow reference: implement equivalent contrastive loss in the common PyTorch harness (NT-Xent, two augmented views, projection head, encoder extraction, seeded training, checkpoint) and document the difference.

Implement and test three cold starts at the **same 40-label initial budget** (and an optional difficult **8-label** diagnostic budget):
- `RandomInit`: uniformly random training IDs without knowing labels.
- `TypiClustInit`: cluster unlabeled pretrained/SSL embeddings, choose typical representatives from distinct groups, without labels.
- `SUPClustInit`: from Ono et al., adapt the clustering/typicality/softmax-weighted boundary score as specified in the original paper. Verify the mathematical implementation with toy numerical tests; no lookups of pool ground truth.

All two-method acquisition comparisons MUST share **the exact same initial 40 queried IDs per seed** so the query-method effect is isolated. The main comparison uses `SUPClustInit + ImageNet transfer` for both methods; separate factorial tests show how much initialization changes results. For an 8-label setting, allow not all eight classes to be represented and handle unseen classes safely; DO NOT secretly use class-stratified sampling with oracle labels. Preserve query histories.

### PHASE 3 — THE TWO PRIMARY ORIGINAL-PAPER ACTIVE-LEARNING METHODS

**Method A: Class-Balanced Active Learning (Bengar et al., WACV 2022).** Reproduce a defensible entropy/class-balancing acquisition variant. Review original `https://github.com/Javadzb/Class-Balanced-AL` code and the paper's exact selection objective/constraints, port to the common ISIC 2019 harness and link source-to-adaptation details. Compute model predictive entropy. Use predicted probabilities/pseudo-class assignments (NEVER hidden pool labels) to encourage acquisition batches that correct observed labeled-set imbalance. If the paper's solver requires Gurobi, use an available open-source optimization formulation (HiGHS/SCIP/OR-Tools) or transparent deterministic greedy approximation; clearly mark implementation as faithful vs adaptation. Prevent model overconfident pseudo-labels from assigning fake perfect balance. Test selected batch size exactly equals budget and predicted vs revealed class distributions are logged separately.

**Method B: CLINICAL (Kothawade et al., 2022).** Implement a submodular mutual-information-based query that targets underrepresented diagnoses using a **reference target set from ALREADY-LABELED rare-class examples** plus informative failures on an approved labeled validation/reference subset. Use feature/gradient similarity as in the actual algorithm; use `submodlib` if compatible and license-permitted, otherwise implement a tested greedy SMI approximation with logged deviation. The original CLINICAL paper runs on ISIC 2018, not ISIC 2019. Do not report its published skin-lesion numbers as our experimental outcomes. **Cold start caveat:** if SUPClust's 40 random/oracle-free labels contain no examples of a rare class, a target set for that class does not exist. Do NOT fish for hidden rare-class labels. Explicitly fall back to supervised available-class targets + diversity/entropy until a minority class is acquired, and record `no_target_for_class`; optionally compare a paid-target-seeding ablation (count each extra query against the budget). Prevent use of test-set misclassifications to create CLINICAL target sets.

Add common reference baselines: `RandomQuery`, `EntropyQuery`, `CoreSet` or `BADGE` (at least one diversity method), and `TypiClust + RandomQuery`. Keep all model/training budgets identical; the only change in the **main A vs B** experiment must be acquisition selection. Run an additional `CB-AL + SUPClust` and `CLINICAL + SUPClust` already captured by common init. The primary comparison is **clean labels and new-label acquisition**.

### PHASE 4 — LONG-TAIL TRAINING, COST SENSITIVITY, AND OPTIONAL NOISY LABELS

**Flexible Sampling**: replicate the paper's core pretraining + anchor/difficulty curriculum in a correctly attributed standalone methodological comparison where its assumptions are permitted. Then build a **clearly named adaptation** `LeakFreeFlexAL`: anchor prototypes estimated exclusively from already labeled items (for represented classes), per-class validation error estimated only where support exists, and unlabeled candidates selected through probability-weighted predicted class + BALD/entropy + diversity. NEVER use true hidden labels to organize the unlabeled pool. Do not present the adapted method as an exact reproduction. Its class-difficulty component can be used *inside the supervised training regimen* as a standalone alternative without claiming new labels were purchased by a sampler.

**Cost-sensitive optional variant**: implement a configurable acquisition cost model and a class-dependent misclassification penalty as **separate concepts**. If only a uniform per-query cost of 1 is available for ISIC, report label counts as actual cost and treat differential annotation cost as a simulation/sensitivity analysis, not measured medical annotation time. Include a rigorously documented strategy such as marginal expected information/utility per expected labeling cost; use predicted class probabilities and known labeled-class counts, not hidden labels. Compare macro-F1, minority recall, labels, and simulated cost. Cite the original cost-aware AL work and mark this weighting as our adaptation/proposal.

**Khanal active label refinement (secondary experiment only)**: run the uploaded MICCAI 2024 code in an isolated setup with training-label corruption p=0.4 and (if resources) p=0.5; add Co-teaching/VOG and active **relabeling** budgets, preserving true clean labels only inside oracle. Log clean vs corrupted training labels and number of expert corrections. This is NOT one of the two primary clean-label AL methods, and its metrics must go in a distinctly labeled noisy-label table. Remove Neptune dependence/API credentials and audit HDF5 bridge. If computationally infeasible, leave a tested, complete script and mark actual results not evaluated rather than fake them.

### PHASE 5 — OUR PROPOSED IMPROVEMENT: `TAILGUARD-AL`

Make a genuinely implementable hybrid and avoid asserting it is novel in the academic literature until the review supports that claim. Call it **TAILGUARD-AL (Transfer-Aware Imbalance and Label-budget-Guided Active Representation and Diversity)** as a *working project name*; use 'proposed adaptation' rather than claiming publication-level novelty.

- **Initialization:** transfer-pretrained ResNet18 feature embeddings + SUPClust (cold start) using unlabeled images only.
- **Training:** class-aware but capped/reliable weighting or effective-number/focal-loss option on **revealed labeled counts**; optionally FlexSampling-style difficulty-aware supervised minibatch curriculum. Handle zero-count classes robustly.
- **Acquisition:** calibrated predictive entropy + soft minority-class acquisition utility from revealed class counts + batch diversity or submodular marginal gain. Example design: `score(x) = wU*norm(entropy(x)) + wM*norm(sum_c p(c|x)/(n_labeled_c+alpha)^gamma * minority_priority_c) + wD*norm(distance_to_labeled_or_batch(x))`. Use validation-only calibration; normalize each score component using only current training-pool predictions; select greedily with explicit batch-diversity penalty. This is a proposed engineering formula, NOT something to attribute verbatim to any cited paper. Avoid using unknown pool labels. Select exactly the allowed batch budget.
- **Class-aware adaptive stopping:** inspect ONLY validation macro-F1, balanced accuracy, and the smallest three class recalls (the minority classes identified *once* from initial training-pool class distribution for analysis, never by unlabeled oracle access inside the strategy). The research harness may compute a once-only distribution audit from ground truth, but strategy can only identify minority classes from revealed labels, approved labeled validation data, or a predefined clinically motivated class set—never from full-pool true counts. Require at least 3 evaluation rounds and a configured minimum budget, then stop only when improvement in macro-F1 over a patience window is below epsilon **AND** a rare-class recall target is met with adequate validation support OR the max budget is exhausted. If a rare class is unrepresented in validation or recall cannot be estimated, do NOT claim it meets the stop requirement; use explicit fallback/max-budget and log reason. Report uncertainty (bootstrap CIs for rare recall when possible). Thresholds/tolerances tuned on validation only, not test.
- **Fair evaluation:** A fixed-budget version of TAILGUARD must run to the same budgets as Methods A/B. Report *additional* early-stop savings separately. Do not compare the early-stopped hybrid at fewer labels directly to a competitor at a different budget without an aligned learning curve.

Ablations (at least): `-no-transfer`, `-random-init`, `-TypiClust-instead-of-SUPClust`, `-no-minority-term`, `-no-diversity`, `-no-difficulty-training`, `-no-class-aware-stop`, `+calibration`, and `+cost-aware` where possible. Use ablations to identify which improvement actually contributes; describe regressions as carefully as improvements.

### PHASE 6 — EXPERIMENT PLAN THAT MUST RUN, NOT JUST EXIST IN YAML

- **Main dataset:** same group-aware ISIC 2019 split across all methods. Native long-tail imbalance is the primary setting. An additional ISIC-2019-only *training-pool* long-tail stress scenario may be made by downsampling training data at a documented imbalance ratio, preserving validation/test untouched; include this only as an explicitly secondary experiment, not as the primary ISIC 2019 result.
- **Main seeds:** `0, 1, 2` for model randomness/initial label selection; shared initial IDs, same available pool, matched splits and label budgets for A/B per seed. Log all RNG and data-loader seeds.
- **Starting labeled budget:** `40` per seed. Cumulative training-label checkpoints: `[40, 80, 160, 320, 640, 1280]`. Optionally also do `8, 16, 24, 32` as separate tiny-budget diagnosis, maintaining strict no-cheat querying. Query fixed batch sizes equal to the differences in checkpoints and do not overquery to fill a predicted class quota.
- **Training:** ResNet18/224, same optimizer/epochs/augmentation/loss for the **strict A vs B acquisition comparison** (choose documented and compute-feasible defaults, e.g. AdamW, fixed epoch ceiling, early stopping only on validation). Pretrain/initialize transfer backbone identically, freeze-unfreeze consistently. Training loss upgrades should be tested in separate controlled arms rather than silently changing optimization for only one primary method.
- **Speed tiers:** a *smoke test* with all methods, one seed, low epoch count and tiny permitted subset; a *full mode* with 3 seeds/all main budgets and paper-style ablations. Smoke outputs must be labeled `SMOKE ONLY`, never included as full evidence. Full runs are the default when real data + suitable GPU are available. Resource-aware runtime optimizations (feature caching, batch inference, batched KNN, sparse similarities, AMP, resume) are encouraged but must preserve experimental fairness. If resource-bound, prioritize **A vs B + Random + Entropy + TAILGUARD** at all six budgets and 3 seeds, and explicitly mark any optional ablation not evaluated.
- **Hyperparameters:** record and lock after validation exploration; no test-set tuning. Implement validation-based checkpoint selection **only**, no maximizing test accuracy across epochs, no cross-experiment cherry-picking. Maintain an untouched final test set for a single final evaluation per completed model/budget after validation checkpoint selection; if test curves are produced after all choices locked, label them accordingly.
- **Metrics per seed/method/budget:** accuracy (micro/top-1), **macro-F1 PRIMARY**, balanced accuracy (macro recall), macro precision, macro recall, weighted-F1 (secondary), one-vs-rest macro AUROC where calculable, per-class precision/recall/F1/support, confusion matrix, and minority-class macro recall/F1 on the 3 least frequent train classes. Handle undefined AUROC or missing classes transparently as `NA` with reason, never zero silently; use `zero_division` policy and report it.
- **Labeling efficiency:** actual labels used per checkpoint, macro-F1 vs labels learning curves, macro-F1 area-under-learning-curve (AULC normalized over a **shared range**), class-acquisition mix by round, label budget to reach 80% and 90% of a full-data supervised **reference performance** (if threshold not crossed write `not reached`), labels to a predeclared absolute macro-F1 target where defensible, and simulated annotation spend if cost-aware module is enabled. The fully supervised reference is a reference, **not guaranteed mathematical upper bound**.
- **Statistical comparisons:** aggregate mean±sample SD across **three** independent seeds with identical budgets and paired seed-level deltas; report 95% uncertainty intervals where feasible, explicitly that n=3 has low power. Paired bootstrap across seeds may be unstable; prefer descriptive effect sizes and label statistical tests exploratory. Report absolute percentage-point improvements and relative percent changes; avoid relative % if denominator is zero or metric NA. Regressions must be reported.
- **Stop comparison:** show each method's result at equal fixed budgets and separately the hybrid's stop budget, stop reason, validation/minority constraints met, test metric at its selected stop checkpoint, and total annotations (including validation).
- **No contamination:** split reproducibility, no train/val/test IDs overlap, no lesion group overlap, no image/lesion near-duplicates across sets where detectable, no pool-label exposure during selection, and no test-driven selection. Generate an automatically checked audit log and fail loudly on a violation.

### PHASE 7 — ENGINEERING AND SCIENTIFIC TESTS (MANDATORY)

Write executable tests using `pytest` covering at least:
1. Dataset IDs/class CSV parsing; eight labels; one-hot correctness; 25,331 count check if data installed.
2. Group-aware split disjointness and reproducibility; no duplicate image or lesion IDs crossing partitions.
3. `Oracle.query()` enforces membership, budget, once-only revelation; an acquisition strategy cannot import/read hidden truth or call ground-truth data directly.
4. A fake oracle that intentionally returns permuted hidden labels **after acquisition** must not change the batch that was selected before reveal; this is an explicit leakage regression test for FlexSampling adapters, CLINICAL, Bengar and hybrid.
5. FlexSampling original `all_labels` acquisition path is not used in live AL; test fails if true labels supplied to it in live mode.
6. Default checkpoint chooser cannot depend on test score; any `cherry` mode is blocked in research runs.
7. Each acquisition scores candidate images and returns **exactly** `min(batch_budget, pool_remaining)` distinct valid IDs, none already labeled.
8. SUPClust/TypiClust cold-start selection works with no labels, including small-`k`, empty-cluster, duplicate-vector, and no-rare-class cases.
9. Class-balancing, cost-aware ranking, diversity, CLINICAL target-missing fallback, and calibration change selection on constructed toy examples as documented.
10. Metric implementation on known confusion matrices, missing-class AUROC, macro vs micro aggregation, AULC common-range interpolation and variance.
11. Early stopping uses only validation metrics, correctly handles insufficient minority support, and cannot read test performance.
12. CLI smoke run, resume, checkpoint hashes, CSV schema compatibility, report plot generation, and final ZIP reopening.

Run `python -m pytest -q`; save `results/tests/pytest_report.txt` and a structured JSON status. Run lint/type checks where tooling available. Fix errors before declaring completion. If the real dataset is blocked, tests that require its bytes may be explicitly skipped with a precise reason; all pure logic and toy-data tests must still pass. Include a test that imports every major module and checks the CLI help commands work on Windows/Linux.

### PHASE 8 — REAL RESULTS, COMPARISON, EXPLANATION, AND IMPROVEMENTS

Generate `results/metrics.csv`, `results/per_class.csv`, `results/queries.csv`, `results/budgets.csv`, `results/summary.csv`, `results/analysis.json` from actual runs with columns for: dataset/split hash, method, init strategy, pretrained checkpoint hash, seed, budget, round, new IDs, cumulative labeled IDs, revealed class histogram, validation metric, test metric, runtime, GPU, checkpoint name, and complete config hash. No blank/misrepresented numerical zeros. If some runs failed, add `results/failed_runs.csv` with exception, step, resource constraint, and recovery instruction.

Create professionally readable visualizations:
- Native class distribution vs labeled-class distribution by AL round.
- Macro-F1 / balanced accuracy / minority macro recall versus **number of labeled training images** with mean±SD ribbons.
- Pairwise A vs B acquisition performance at matched budgets and label savings curves.
- Per-class precision/recall/F1 grouped bars, confusion matrices for each main method at 1280 labels, calibration/reliability if available.
- AULC table, labeling efficiency, cost simulation (clearly labeled), stopped budget and stop reasons.
- Component ablation chart with delta vs full TAILGUARD.
- Examples of randomly sampled failures chosen **from final evaluation only**, with IDs/redacted privacy and no clinical claims; use actual images only if licensing permits internal project presentation.

Write an evidence-grounded discussion that explicitly answers: Which primary AL method works best? At which label budgets? Which classes benefit or suffer? Does SUPClust or transfer learning help at 40 labels? When does CLINICAL lack rare references? Does diversity reduce redundancy? Does class-aware stop save annotations without unacceptable minority recall losses? Is high overall accuracy concealing minority-class failure? Did our hybrid improve both macro-F1 and minority performance? Which proposed modifications (calibration, balanced loss, capped minority weighting, annealed acquisition, SSL, query cost) should be retained, rejected, or evaluated later? **Do not predict victory in advance.**

### PHASE 9 — RESEARCH LITERATURE REVIEW AND FINAL ARTIFACTS

Produce **a properly cited academic literature review** (`docs/literature_review.md` and PDF) with: abstract, introduction to AL and class imbalance, why naive uncertainty sampling may neglect rare diagnoses, categories (class-aware acquisition, cost-sensitive/annotation-cost AL, diversity-based AL, targeted imbalance-aware queries, cold-start initialization, label-noise cleaning), evidence table of **at least 10 original research papers** with accurate author/year/venue/DOI/URL, method mechanics, applicability to ISIC 2019, weaknesses, true-label requirements, comparison of the selected two, evidence about transfer learning, and our proposed improvement. Quote sparingly; paraphrase accurately. Distinguish review findings from new experiments. Provide in-text citations (`[Bengar et al., 2022]`) and compiled bibliography with resolvable URLs. Never fabricate a DOI or performance number.

Produce **`results/final_report.md` and PDF**, with:
1. Abstract, problem statement exactly aligned to latest slide, research questions/objectives.
2. Project-specific description of the exact ISIC 2019 data and eight class counts, group-aware split decisions, imbalance measures.
3. Literature evidence and method choice rationale: Bengar vs CLINICAL, with source citations.
4. System architecture/data flow diagram and original/adapted modules labeled.
5. Equations/pseudocode for both AL methods, transfer/SUPClust init, and TAILGUARD acquisition + stopping.
6. Detailed experimental protocol and environment; exact seeds, budgets, metrics, hyperparameters, validation rule, annotation ledger.
7. **Primary A vs B comparison table** at budgets `40, 80, 160, 320, 640, 1280`: Accuracy, macro Precision, macro Recall, macro F1, Balanced Accuracy, minority recall/F1, AUROC, AULC where defined; mean±SD 3 seeds; test checkpoint selection audited.
8. Baseline, hybrid, ablation, full-data reference, and separate Khanal noisy-label tables; NO cross-paper score mixing.
9. Statistical uncertainty, labeling efficiency, failure cases, non-improving outcomes, clinical/domain limitations and ethical/distribution notes.
10. Detailed answer to 'how to improve these methods' with demonstrated improvements vs **future, untested proposals** separated.
11. Reproduction instructions, exact commands, evidence filenames, and bibliography.
12. Explicit **COMPLETE / PARTIAL / BLOCKED** status matrix for each module and each actual experiment.

Produce a **clean editable PowerPoint** `presentation/ISIC2019_Active_Learning_Research.pptx` for a university viva: 12–18 readable slides, title, problem statement, objectives, ISIC classes and imbalance, 10-paper literature evidence, two chosen methods, cold start and transfer, architecture flow, fair experimental setup, real metric comparison and plots (or clearly labeled NOT EVALUATED placeholders where results unavailable), our improvement, ablation results, strengths/weaknesses, conclusion and citations. Use a polished cohesive design, large text, neat diagrams. Create `docs/viva_guide.md` with explanation of every algorithm, metric, AL cycle, why class imbalance matters, why macro-F1 is primary, likely viva questions/answers, and short project demo script.

Create a **genuinely usable Streamlit dashboard** launched by `streamlit run dashboard/app.py`: overview, source audit, method comparison filter, seed/budget selection, interactive charts, query-label distribution, minority-class performance, stopping audit, experiment status, downloadable CSV/PNG. It must read real files; if no results exist, show an informative `No verified experiment results yet` banner and instructions, not invented/demo metrics in the results pages. It is a research dashboard, not a medical diagnostic app. Optional inference tab may accept a lesion image to show eight class scores only after a verified checkpoint exists; prominently label 'Research demonstration, not clinical diagnosis'.

### PHASE 10 — AUTONOMOUS EXECUTION ORDER AND COMPLETION GATES

Execute in this order and write durable progress after each gate:

**G0 Sources:** identify all uploaded files; audit ZIP contents, PDF papers, slide, and audio where possible; verify original papers/repositories and freeze scope in `docs/requirements_traceability.md`.
**G1 Data:** acquire and validate REAL ISIC 2019 images/CSV, groups, splits, histograms; if blocked, preserve blocker + downloader and keep coding.
**G2 Architecture:** implement full importable common codebase, sealed oracle, dataset, models, transfer/SSL, initialization, acquisition, stopping, metrics and config.
**G3 Methods:** implement Bengar vs CLINICAL, baselines, SUPClust/TypiClust, leak-free FlexSampling adapter, TAILGUARD; noisy Khanal secondary separately.
**G4 Tests:** run tests, fix leakage and cherry-selection, run static checks and CLI dry-run.
**G5 Smoke:** when real ISIC data exists, run end-to-end one seed/low-budget smoke for every main arm; inspect real metrics/logs and repair failures. Otherwise only synthetic *unit* tests and mark smoke NOT EVALUATED.
**G6 Full:** train and evaluate the predeclared 3-seed, 6-budget priority matrix, then ablations/secondary experiments subject to resources. Execute—not simply schedule. Resume failed sessions from checkpoints.
**G7 Analysis:** run aggregation, paired comparisons, minority analysis, uncertainty, plots and verifiable comparison tables; detect invalid seeds/splits/budgets and exclude only with documented reasons.
**G8 Documents/UI:** write literature review, report, citations, PowerPoint, viva guide, dashboard, and README from actual findings/status.
**G9 Quality:** rerun tests; check every citation resolves, generated PDFs/PPTX open, Streamlit starts, scripts work, and no unverified number appears in the final report.
**G10 Package:** remove raw ISIC images, user secrets, caches, virtual environments and oversized outputs not necessary; preserve source, configs, manifests, sample usage, final results, plots, logs, reports, original uploaded **papers** if redistribution rights permit, and instructions to supply input data. Produce `outputs/ISIC2019_Imbalance_AL_COMPLETE.zip`; reopen and verify expected paths and hashes. Include `MANIFEST.sha256` and `RUN_STATUS.md` inside and show download/export path.

For every gate, write one row in `PROGRESS.md` with time, status, command executed, evidence file path, and blocker if any. Never quietly mark a gate done based on file existence alone; validate content and functionality.

### FINAL MESSAGE TO ME (MANDATORY FORMAT)

At the very end show:
- **Actual completion status**: counts of gates complete / partial / blocked and whether real ISIC 2019 training was actually executed.
- **Verified two-method comparison**: a table of Bengar vs CLINICAL at each available matching training-label budget with real accuracy, macro-F1, balanced accuracy, precision, recall, and minority recall, clearly showing mean±SD for 3 seeds where available, or `NOT EVALUATED` if not run.
- **Our improved model vs each primary method**: improvement or regression in metric points; label efficiency, stopping savings and caveats; distinguish tested from proposed.
- **Literature coverage**: exact count of original papers with bibliography file.
- **Deliverables**: project ZIP absolute path and download/export link if available; location of report, review, PPTX, dashboard, reproducibility commands.
- **Reproducibility**: 3–5 exact setup/run/test commands for the user's detected OS.
- **Transparency**: list anything not actually finished and exact blocker/recovery steps; don't conceal lack of data/GPU.

Do the work now. **No stopping at planning, design, or placeholder code; actual execution and verification are the acceptance criteria.**
