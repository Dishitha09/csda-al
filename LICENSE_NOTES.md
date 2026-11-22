# License and attribution audit

Audit date: 21 November 2026. These are observations about supplied files and planned distribution, not a legal opinion or a grant of rights.

| Supplied scope | Actual notice inspected | Integration rule |
|---|---|---|
| Active Label Refinement | MIT, copyright 2024 Bidur Khanal | Research citation; preserve notices if any substantive source is later adapted. No reference sources copied in Stage A. |
| Google SimCLR | Apache License 2.0; source headers credit SimCLR Authors | Original PyTorch implementation from the method, no TensorFlow source dependency; track any direct adaptation separately. |
| TypiClust root | MIT, copyright 2022 Avihu Dekel and Guy Hacohen | Supporting method/reference citation; do not assume this covers every subtree. |
| TypiClust `deep-al` | MIT, copyright 2022 Akshay L Chandra and Vineeth N Balasubramanian | Distinct nested notice; preserve it if adapting source. |
| TypiClust `scan` | **CC BY-NC 4.0**, confirmed from its full LICENSE | The input pack's broad MIT summary is incomplete. Do not copy this subtree into CSDA-AL. |
| Flexible Sampling | README and pyproject assert MIT, but no LICENSE/LICENCE/COPYING file supplied | Read-only reference; original concept-based implementation. Do not treat metadata alone as verified complete reuse provenance. |
| Unofficial SUPClust starter | No LICENSE file supplied; explicitly not author code | Independently implement paper equations; do not redistribute the starter in the integrated ZIP. |
| Three supplied papers | User-provided literature; redistribution permission not established by this audit | Cite identities, pages and equations; exclude PDFs from the original project ZIP. |
| ISIC 2019 | Official data page lists CC-BY-NC for 2019 and required dataset citations | Dataset remains user-supplied and excluded from the final architecture ZIP. Real experiments must preserve its attribution requirements. |

The original resources, including their licenses, remain intact in the separate extracted resource pack. The four standalone uploaded repository ZIPs were checked against their corresponding bundled trees: 228 matching files, zero mismatches. This establishes snapshot correspondence, not current upstream provenance or a commit identity.

No project-wide license is assigned at Stage A. The research owner can choose one before public distribution. Dependency licenses and any direct source reuse must be included appropriately regardless of that choice. Dependency installation is a later-stage task; this audit does not label uninstalled dependencies as reviewed.

See [research citations](docs/RESEARCH_TRACEABILITY.md), [SUPClust license details](docs/audits/SUPCLUST_AUDIT.md), [Flexible Sampling audit](docs/audits/FLEXIBLE_SAMPLING_AUDIT.md), and [Refinement/SimCLR audit](docs/audits/REFINEMENT_SIMCLR_AUDIT.md).
