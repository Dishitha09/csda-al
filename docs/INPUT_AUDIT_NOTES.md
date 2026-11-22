# Initial audit of supplied assets (22 November 2026)

- `AL imbalance class(2).zip`: three original papers (Ju 2204.03161, Khanal 2407.05973, Ono 2403.03741), not data or training results.
- `FlexSampling-main(1).zip`: 56 ZIP entries, 23 `.py` files, README, YAML configs, pre-defined NumPy splits. Public original code exists at https://github.com/PyJulie/FlexSampling. This archive has meaningful code, but its suitability for leak-free active learning is **not established**.
- Critical acquisition leak: `flexsampling/core/anchor.py` selects class anchors using labels across the full training dataset; `flexsampling/core/curriculum.py` consumes `all_labels` in `query`; `flexsampling/core/pipeline.py` obtains `_all_labels` from all `train_dataset` records and sends them into acquisition. Using these methods unchanged on a simulated hidden-label pool would cheat.
- Critical test selection issue: `examples/run_flexsampling.py` defaults `FLEX_SELECT` to `cherry`, which tracks maximal test score across validation-improvement checkpoints. Must use validation-only selection. The script also hardcodes a Windows F-drive data directory.
- `TypiClust-main(1).zip`: 169 entries, 76 `.py` files, research implementation and documentation. Audit before reuse.
- `simclr-master(1).zip`: 41 entries, contains TensorFlow-based SimCLR code; use as paper reference or documented PyTorch port.
- `imbalanced-medical-active-label-cleaning-main(1).zip`: 14 entries, 6 `.py` files. Khanal's MICCAI 2024 project is active LABEL REFINEMENT under noise, not ordinary pool-based acquisition. Compare only in a separate noisy-label protocol.
- `WhatsApp Audio...mp4`: AAC audio track, around 303.4 s; not transcribed in this audit. Agent should attempt transcription and note any extra requirements.
- The slide says transfer-learning cold start and class-aware stopping. The older 9-page PDF explicitly excludes them. Latest user request and slide take precedence; the old approach becomes ablation.
- No ISIC 2019 image dataset was included in any of the inspected ZIP archives. Therefore no real experiment metrics were calculated during prompt preparation.
- Choose two true imbalance-aware query methods as primary (Bengar Class-Balanced AL vs CLINICAL) and add Ju's Flexible Sampling/SUPClust as modular, properly identified components.

The findings above describe inspected files, not completed execution of the intended research project.
