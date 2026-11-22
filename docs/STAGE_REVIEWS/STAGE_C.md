# Stage C: models and feature layers

Implemented original PyTorch ResNet-18 backbone, SimCLR two-layer projection head and stable NT-Xent averaged over 2N anchors. Features come from the 512-dimensional backbone before projection. Dual augmentations operate independently and conservative color/crop settings are configurable. The eight-logit classifier accepts strict SSL encoder transfer; optional supervised pretrained initialization is cache-only and never downloads weights automatically.

SupervisedTrainer accepts acquired TRAIN datasets only. Head, layer4 and all-backbone stages control parameter groups and frozen BatchNorm buffers; set_freeze_stage provides staged unfreezing. Weighted CE is opt-in and derives frequencies only from acquired labels. Feature stores validate unique aligned IDs, finite arrays and normalization; caches bind identities and payload checksums. Checkpoints include optimizer/scheduler when supplied, class order, metadata and Python/NumPy/PyTorch RNG states. Inference restores caller module modes.

Source relevance: SimCLR's encoder/projector/contrastive structure and SUPClust's SSL feature space are preserved; TensorFlow source is not a runtime dependency. Conventional anchor-mean NT-Xent is half Google's sum of directional means, a documented reduction choice.

The initial focused model/feature run passed 17 tests. Final full suite and all imports passed; exact commands/output are retained under ../test_logs/. Tests include real synthetic optimizer steps, gradients, save/reload, cache corruption/staleness and frozen encoder parameters/BatchNorm buffers. Separate SSL CLI dry run performs exactly two generated-image optimizer steps; no real ISIC representation metric was measured.

Limitations: augmentation validity and representation quality require later lesion experiments. The synthetic AL workflow uses an untrained encoder for wiring only. Freezing mode is configured per run and can be changed via the trainer API; no automatic scientific schedule is claimed tuned.
