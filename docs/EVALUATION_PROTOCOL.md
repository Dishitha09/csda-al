# Evaluation protocol

The evaluation package calculates observations from supplied predictions. It does not contain ISIC results, trained checkpoints, a benchmark claim, or generated scientific curves. Synthetic fixtures exercise correctness only.

## Data and access policy

Use the diagnostic order `MEL, NV, BCC, AK, BKL, DF, VASC, SCC`. `evaluate_predictions` accepts one-dimensional integer predictions or an aligned `N x 8` probability matrix. Probabilities must be finite, nonnegative, at most one, and sum to one per row. Ties use the first class in the declared order. The caller must align prediction rows to target image IDs before passing arrays; this pure evaluator cannot reconstruct alignment from bare arrays.

Validation targets belong to evaluation and class-difficulty estimation. They must never be included in acquisition context, candidate metadata, feature caches, or trainer inputs. Test targets are used only after final method, checkpoint, split and configuration choices are locked. Test records are never used for SSL, class difficulty, acquisition, checkpoint selection or threshold tuning.

`EvaluationLock(checkpoint_id, config_fingerprint, split_fingerprint)` declares the final identities. Test calls must pass that lock and exactly matching identity strings. A mismatched or unlocked call fails. These are audit controls: the caller must hash the actual files, and a Python object cannot prevent someone from editing a split or repeatedly examining test results outside the application.

## Metric definitions and missing support

`classification_metrics(y_true, y_pred, num_classes=8)` returns only JSON-safe Python values. Confusion matrix rows are true classes; columns are predicted classes. Each class has precision, recall, F1, true support, predicted count and an explicit supported flag.

- Accuracy is correct examples divided by evaluated examples.
- Balanced accuracy is mean recall over classes with true evaluation support. Its denominator is disclosed in `balanced_accuracy_policy` and the support array.
- Macro precision/recall/F1 average all eight declared classes. Unsupported or never-predicted cases use zero division equal to zero. Thus fixed-eight macro recall can differ from balanced accuracy when a class has no true support.
- Micro precision/recall/F1 equal accuracy for this exclusive single-label task.
- Weighted precision/recall/F1 use true evaluation support as weights.
- An entirely empty evaluation returns accuracy and balanced accuracy as `null`, with zero support and explicitly zero-valued precision/recall/F1 conventions. It does not manufacture a measured accuracy of zero.

Interpret missing support as missing evidence. A zero-valued convention is not proof a model failed on an absent class. Do not claim all eight classes were learned because an eight-logit classifier was fitted to a tiny label set.

## Annotation and compute accounting

`EvaluationResult` records `label_budget` (unique acquired training annotations) and `validation_annotation_cost` separately. `total_annotation_cost` is their sum. A preexisting externally labeled validation set retains its disclosed cost; it does not become free because it was labeled before the run. Evaluation support describes rows evaluated and is not automatically a new annotation charge. The engine's ledger owns unique-image accounting.

Report all curves using a declared x-axis: acquired training labels, or total training-plus-validation annotation cost. Do not compare one method's training-only curve to another method's total-cost curve. Test annotation availability is separately documented as evaluation infrastructure rather than an acquisition allowance. Log runtime, optimizer steps and hardware independently from annotation cost.

## Head, middle and tail reporting

`frequency_groups(acquired_training_labels, head_min=20, middle_min=5)` uses cumulative acquired labels only. Head classes have at least `head_min` acquired labels, middle classes have at least `middle_min` and fewer than `head_min`, and tail classes have fewer than `middle_min`. Unobserved classes are listed separately and included in tail. These configurable thresholds are CSDA-AL reporting conventions, not thresholds claimed from a paper.

`head_middle_tail_metrics` returns group support, fixed-group macro F1, and supported-class balanced accuracy. Empty groups have `null` metrics; groups with no evaluation support have `null` balanced accuracy. Group membership can change with label budget and must therefore be saved with each report. Never calculate these groups from sealed full-pool truth. A future fixed external grouping requires a separately declared and costed provenance.

## Learning curves and label efficiency

`normalized_aulc(budgets, scores, budget_interval=None)` integrates piecewise-linear scores by trapezoids and divides by interval width. It requires finite scores in `[0,1]`, at least two aligned points and strictly increasing nonnegative budgets. No sorting, deduplication or extrapolation occurs silently. Predeclare the comparison metric and common budget interval.

`compare_aulc({method: (budgets, scores)}, budget_interval=None)` uses one provided interval or the common overlap of all observed curves. It rejects an empty overlap. Save the reported interval with every comparison, since different intervals can change the result.

`label_savings(baseline_budgets, baseline_scores, candidate_budgets, candidate_scores, target_score)` locates the first piecewise-linear target crossing for each curve. Positive savings favor the candidate; negative savings are retained. Targets must be predeclared. An unattained target returns a descriptive status and `null` savings; no future attainment is extrapolated. An interpolated count is an estimate rather than an observed integer annotation count. Nonmonotonic curves use the first crossing, without replacing observations by an undisclosed best-so-far envelope.

Repeated seeds, uncertainty intervals and statistical comparisons require actual later experiments. No confidence interval is invented from one synthetic run.

## Exports and plotting

`export_result(result, output_dir, prefix='evaluation', synthetic=False)` writes a canonical versioned JSON plus per-class and confusion-matrix CSV views. Each file is atomically replaced; the three-file set is not one filesystem transaction. Use unique per-round directories to avoid mixing versions. Set `synthetic=True` for all synthetic demonstrations; the JSON then says `synthetic_demonstration_no_real_training`.

`plot_learning_curves(curves, output_path, synthetic=True)` saves a standalone PNG/PDF/SVG from supplied points and visibly labels a synthetic plot `SYNTHETIC DEMONSTRATION - NO REAL TRAINING`. The plotter creates no observations. Use the default non-synthetic mode only for actual observations with provenance. Class targets and predictions are not exported by this helper, only aggregate results.

Validation example (synthetic, no real training):

```python
from csda_al.evaluation import evaluate_predictions, export_result

result = evaluate_predictions(
    [0, 0, 1, 1], [0, 1, 1, 2],
    label_budget=8, validation_annotation_cost=4,
    checkpoint_id="synthetic-fixture",
)
export_result(result, "outputs/synthetic_evaluation", synthetic=True)
```

Tests use hand-computed arrays for confusion, class support, averaging differences, target crossings and area calculations. These fixtures are not ISIC metrics and cannot support a model-quality claim.
