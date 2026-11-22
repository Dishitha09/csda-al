# Research limitations and experimental boundaries

CSDA-AL is an original integration for later research evaluation. Executable components and passing synthetic tests establish software behavior; they do not establish improved lesion recognition, calibration, clinical utility, fairness or label efficiency. No full ISIC 2019 training, clinical validation, published-result reproduction, or real representation quality measurement is included.

## Source relevance and deviations

SUPClust supplies a label-free cluster/typicality/boundary mechanism, with clusters representing feature geometry rather than diagnostic classes. SimCLR supplies the contrastive representation principle; the implementation uses PyTorch rather than the supplied TensorFlow runtime. Flexible Sampling supplies learning-difficulty-aware curriculum ideas. Its known-label prototypes and per-class buckets cannot be applied to a sealed unlabeled pool; CSDA-AL instead combines model-predicted membership with validation-derived class difficulty and normalized uncertainty. This score fusion is a proposed adaptation, not an algorithm or result attributed to that paper.

Single-label per-true-class accuracy in Flexible Sampling is the same statistic as recall. CSDA-AL's prior smoothing, unsupported-class policy, predicted-membership weighting, entropy fusion and optional diversity reranking are explicit additional choices. The Active Label Refinement paper supplies useful acquisition-loop and baseline design ideas, while its actual task corrects existing noisy labels. Co-teaching/VOG label cleaning and class-aware stopping remain outside the primary project.

## Dataset integrity and cost

ISIC 2019 is the main dataset, with eight known diagnostic classes in a fixed order. Public training labels are simulated hidden oracle labels during an AL run. A software-private mapping is an engineering boundary, not cryptographic protection against a user modifying code. Tests must also check that changing sealed labels leaves current selections unchanged.

Split metadata may lack patient identifiers. Grouping by lesion/patient when present reduces related-image leakage, but absent identifiers prevent a guarantee of independent patients. Label-free group splitting can leave validation classes unsupported. Introducing stratification based on the entire sealed training-label vector would consume privileged label information and is excluded from default acquisition setup.

A zero-acquired-training-label start does not imply zero total annotation cost when a labeled validation set is later used. Validation counts, provenance and cost are disclosed separately. Repeatedly adapting acquisition to the same validation set can overfit it even though candidate truth remains sealed. Reserve test data for final locked assessment, and document any method tuning on validation.

## Representation and selection uncertainty

SSL does not guarantee imbalance removal or preservation of all diagnostic cues. Crop strength, color jitter, grayscale and blur can remove lesion boundaries or alter relevant colors. The selected conservative/default transforms require later sensitivity analysis; they are engineering defaults rather than clinically validated invariances.

Small budgets may contain only a subset of diagnoses. An eight-logit classifier cannot create absent supervision. Estimated class difficulties are noisy when validation support is low; an explicit prior for an unsupported class is an assumption, not measured evidence. Report support, smoothing and prior-used flags, and compare sensitivity to these settings.

Predicted-membership weighting can suppress a rare-class image confidently misclassified as a majority class. Entropy and BALD measure forms of model uncertainty, not clinical importance or the true value of annotation. Dropout disagreement is meaningful only under its declared model/training approximation; inference-only layer insertion is not a substitute for a trained stochastic model. Diversity does not guarantee diagnostic-class coverage.

K-Means and nearest-neighbor geometry depend on feature normalization, encoder identity, preprocessing, temperature and random seed. Degenerate embeddings require documented fallbacks; a fallback should not be reported as exact paper reproduction. Exact neighbors can be expensive on large pools, and approximate search can change selections. Save backend and configuration with each query.

## Evaluation interpretation

Balanced accuracy over supported classes can look high while an absent evaluation class has no evidence. Fixed-eight macro metrics make unsupported-class conventions explicit but mix missing evidence with a numerical zero convention. Always read both with support and confusion matrices. Acquired-frequency head/middle/tail groups can change between rounds; comparison reports must save group membership.

Normalized area and target-based savings depend on the shared budget interval, chosen metric, validation cost and interpolation. A target not reached is unresolved, not an invitation to extrapolate. A synthetic eight-class smoke test is deliberately simple and cannot predict real performance or realistic annotation difficulty.

Future experimental work should predeclare budgets, seeds, baselines, score fusion, split policy, annotation-cost treatment and final targets; report variability over actual runs and investigate predictive bias. Those experiments require separate authorization and real data. This architecture does not provide a diagnostic tool for patient care.
