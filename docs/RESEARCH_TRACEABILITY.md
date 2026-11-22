# Research-to-implementation traceability

> Current implementation status (21 November 2026): Stages B-F were subsequently authorized together and implemented. The Stage A design below records original intent; current APIs and any differences are documented in [MODULE_GUIDE.md](MODULE_GUIDE.md) and the completion report.

This map connects research evidence to implemented modules. No measured scientific improvement or exact experimental reproduction is claimed. Actual software checks are in the stage reviews and completion report.

| Source and evidence | Proposed modules | Required evidence in later stages | Distinction/limitation |
|---|---|---|---|
| SUPClust, Section 3, Eqs. (2)-(3), printed p. 3; four steps p. 4 | `acquisition/supclust.py`, `features/*` | Independent hand-calculated weights/SUP; first/subsequent rounds; occupied clusters; tiny/duplicate pools | Clusters are geometric proxies, not known diagnostic classes. |
| SUPClust, Eq. (1), p. 2; TypiClust within-cluster caller | `features/nearest_neighbors.py`, `acquisition/typiclust.py` | Ordinary Euclidean neighbor distances and explicit self exclusion | Neighbor scope/count, rounding and singleton policy are declared conventions. FAISS squared L2 must be converted. |
| SUPClust, Section 4.1, p. 4 | `models/simclr.py`, `models/classifier.py`, `training/finetune.py` | Backbone-before-projection features; frozen linear and fine-tuned modes | Supports the architecture, not an ISIC performance claim. |
| SUPClust, Section 4.2, p. 4 | `acquisition/supclust.py` ablations | `without_typicality` and seeded-random `without_SUP` tests | Most-typical selection is a distinct TypiClust baseline. |
| Flexible Sampling, Section 2.2, Eqs. (2)-(3), pp. 4-5 | No primary reuse of prototype anchors | Leakage review rejects hidden-class buckets | Known-label anchor construction is replaced by SUPClust. |
| Flexible Sampling, Section 2.3, Eq. (4), p. 5 | `imbalance/class_difficulty.py` | Correct true-class recall, support and explicit missingness/smoothing | Per-class accuracy equals recall here; smoothing/prior is a project extension. |
| Flexible Sampling, Section 2.3, Eq. (5), pp. 5-6 | `acquisition/bald.py` | Positive MI, repeatable stochastic passes, BatchNorm isolation | Supplement p. 12 has reversed pseudocode sign; main equation governs. BALD is optional. |
| Our proposed unlabeled adaptation | `imbalance/probability_weighting.py`, `acquisition/difficulty_aware.py` | `p @ difficulty`; normalized entropy and difficulty fusion; alpha endpoints | Prediction membership and score fusion are not an algorithm reported in the paper. |
| Active Label Refinement, Section 2.4, p. 5 | `engine/active_loop.py`, `engine/budget.py`, acquisition protocol | Fixed budget, oracle transaction, cumulative labeled set, deterministic histories | Changes relabeling of noisy examples into new-label acquisition. |
| Active Label Refinement supplied strategies/coreset modules | `acquisition/{entropy,coreset,random}.py` | Hand ranks, greedy coverage updates, empty-seed/duplicate handling | Write original robust code; no target-bearing acquisition loader. |
| Active Label Refinement, Section 2.3, Eq. (1), pp. 4-5 | Documentation only | Verify no VOG/Co-teaching/noise branch in primary engine | Explicitly excluded from the model. |
| Chen et al. SimCLR; supplied Google `objective.py`, `model_util.py`, `model.py` | `models/simclr.py`, `training/ssl_train.py`, `data/transforms.py` | NT-Xent identities, independent two-view behavior, one CPU step, checkpoint equality | PyTorch reimplementation; standard `2N` anchor mean differs by factor two from Google's directional-sum reduction. |
| Official ISIC 2019 dataset/challenge description | `data/metadata.py`, `data/split.py`, `evaluation/*` | Fixed named eight-class mapping, grouped manifest, unknown-class rejection | Eight training classes; official challenge test adds an outlier category. Closed-set study declared explicitly. |

## Bibliographic identities

1. Yuta Ono, Till Aczel, Benjamin Estermann and Roger Wattenhofer. **SUPClust: Active Learning at the Boundaries**. ICLR 2024 Workshop on Practical Machine Learning for Low Resource Settings. Supplied PDF: arXiv:2403.03741v1, 6 March 2024. [Paper](https://arxiv.org/abs/2403.03741).
2. Lie Ju, Yicheng Wu, Lin Wang, Zhen Yu, Xin Zhao, Xin Wang, Paul Bonnington and Zongyuan Ge. **Flexible Sampling for Long-tailed Skin Lesion Classification**. MICCAI 2022, pp. 462-471; DOI `10.1007/978-3-031-16437-8_46`. Supplied PDF: arXiv:2204.03161v2, 28 May 2025. [Paper](https://arxiv.org/abs/2204.03161).
3. Bidur Khanal, Tianhong Dai, Binod Bhattarai and Cristian Linte. **Active Label Refinement for Robust Training of Imbalanced Medical Image Classification Tasks in the Presence of High Label Noise**. MICCAI 2024. Supplied PDF: arXiv:2407.05973v3, 24 October 2024. [Paper](https://arxiv.org/abs/2407.05973).
4. Ting Chen, Simon Kornblith, Mohammad Norouzi and Geoffrey Hinton. **A Simple Framework for Contrastive Learning of Visual Representations**. ICML 2020, PMLR 119:1597-1607. [Primary publication](https://proceedings.mlr.press/v119/chen20j.html), [Google code](https://github.com/google-research/simclr).
5. Guy Hacohen, Avihu Dekel and Daphna Weinshall. **Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets**. ICML 2022. Supporting TypiClust implementation: [repository](https://github.com/avihu111/TypiClust). Supplied source snapshot, not a verified current commit.
6. ISIC. [2019 challenge description](https://challenge.isic-archive.com/landing/2019/), [official data and required dataset citations](https://challenge.isic-archive.com/data/). Official pages checked 21 November 2026. Dataset attribution requirements must be carried into later real experiments; no data is redistributed here.

Detailed source symbols, line numbers, revision caveats and actual reading commands are in [SUPClust audit](audits/SUPCLUST_AUDIT.md), [Flexible Sampling audit](audits/FLEXIBLE_SAMPLING_AUDIT.md), and [Refinement/SimCLR audit](audits/REFINEMENT_SIMCLR_AUDIT.md). Line numbers identify the immutable supplied snapshots, not online HEAD versions.
