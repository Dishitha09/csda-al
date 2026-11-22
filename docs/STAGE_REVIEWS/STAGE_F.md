# Stage F: verification and packaging

The user authorized Stages B-F together with "do all stages now". All required implementation stages are complete; no further approval pause is inferred. NO REAL ISIC TRAINING.

Actual final source verification: **131 tests passed**, Ruff passed, configured Mypy passed, all CLI checks passed. Mypy checks project code while skipping large third-party implementation type trees; runtime shapes/semantics are covered by tests. The external-data inspection command correctly exits 2 with the missing-image-directory message because ISIC is not included.

| Check | Outcome | Actual / expected exit |
|---|---|---|
| pytest | PASS | 0 / 0 |
| ruff | PASS | 0 / 0 |
| mypy | PASS | 0 / 0 |
| inspect_missing_data | PASS | 2 / 2 |
| ssl_dry_run | PASS | 0 / 0 |
| feature_dry_run | PASS | 0 / 0 |
| cold_start | PASS | 0 / 0 |
| active_loop | PASS | 0 / 0 |
| active_loop_resume | PASS | 0 / 0 |
| evaluate_help | PASS | 0 / 0 |
| all_imports | PASS | 0 / 0 |

Exact executable/arguments/stdout/stderr are retained in ../test_logs/; pytest.xml lists individual tests. Source output metrics/checkpoints are generated from synthetic fixtures only and are excluded from the ZIP. No fictitious model checkpoint is shipped.

The package builder includes all source, configs, tests, docs, requirements/lock, Mermaid source and completion report under csda_al/. ARCHIVE_MANIFEST.json records bytes/SHA-256 for each payload member. It rejects missing required artifacts, excludes references/images/weights/caches/temp outputs, checks CRCs, extracts safely and verifies member hashes/imports. Extracted test verification is recorded after the first complete package build; final packaging rechecks unchanged source and final documentation.

Limitations: the CPU architecture is verified, not scientific performance, full-data throughput or clinical validity. The risk register and research limitations remain applicable. See ARCHITECTURE_COMPLETION_REPORT.md at the repository root for the final artifact status.

Actual isolated extracted run: **131 passed, 14 third-party deprecation warnings in 39.19s**. CRC/member hashes/extracted hashes/imports passed. Raw output is retained in ../test_logs/archive_extracted_tests.txt. Final documentation repack is checked against the tested source/config/test manifest; code is unchanged.
