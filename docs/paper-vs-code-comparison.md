# Paper vs. Implementation Comparison

## Purpose

This document compares the 2016 DeepConvLSTM paper by Ordóñez and Roggen with the modern PyTorch implementation used in this reproduction. The goal is to separate architectural agreement from differences in data processing and experimental protocol. Until the differences below are resolved or explicitly accepted, this project should be described as a reproduction attempt rather than an exact reproduction.

## Sources inspected

- Original paper: Ordóñez and Roggen, “Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition” (2016), https://pmc.ncbi.nlm.nih.gov/articles/PMC4732148/
- Original authors' repository: https://github.com/STRCWearlab/DeepConvLSTM
- Modern PyTorch implementation: https://github.com/STRCWearlab/DeepConvLSTM_py3
- Files inspected in the modern repository: `DeepConvLSTM_py3.py`, `datasets.py`, and `preprocess_opportunity.py`.

## Comparison table

| Area | Original paper / original setup | Modern PyTorch repository | Assessment |
|---|---|---|---|
| Network architecture | Four convolutional layers followed by two LSTM layers and a classifier | Four `Conv2d` layers, two-layer `nn.LSTM`, linear classifier | Core architecture is aligned |
| Convolution | 64 filters per convolution, temporal kernel size 5, no pooling | 64 filters, kernel `(5, 1)`, no pooling | Core convolution configuration is aligned |
| Recurrent layers | Two LSTM layers, 128 units | `nn.LSTM(..., hidden_size=128, num_layers=2)` | Aligned at the high level |
| Dropout / initialization | Dropout and orthogonal initialization are part of the reported setup | Dropout 0.5; utility initializes recurrent and linear/convolution weights orthogonally | Broadly aligned; verify exact layer-by-layer initialization details if strict fidelity is required |
| Input channels | 113 sensor channels for this OPPORTUNITY setup | 113 sensor channels | Aligned |
| Windowing | Paper describes windows of about 500 ms with a 250 ms step (at 30 Hz, approximately 15 samples and 7–8 samples per step) | Defaults: 24 samples and 12-sample step | Different: modern window is longer and advances farther per step |
| Sensor normalization | Per-channel normalization to a bounded range using predefined thresholds in the original preprocessing | Computes mean and standard deviation from training data, then applies `(x - mean_train) / std_train` to all splits | Different preprocessing; can change optimization and results |
| Train/validation/test assignment | Paper's protocol: S1 ADL/Drill for training; S2/S3 Drill and ADL1–ADL2 for training; S2/S3 ADL3 for validation; S2/S3 ADL4–ADL5 for testing | Training: S1 Drill and ADL1–ADL4 plus S2/S3 Drill and ADL1–ADL3; validation: S1 ADL5; test: S2/S3 ADL4–ADL5 | Different validation split and therefore a different experimental protocol |
| Optimizer | RMSProp | Adam by default | Different |
| Initial learning rate | Paper setup reports 0.001; implementation details should be checked against the exact experiment being reproduced | 0.001 by default | Same initial value, different optimizer |
| Learning-rate schedule | Paper's stated RMSProp/decay configuration | StepLR: default step size 100 epochs and gamma 0.1 | Different schedule semantics |
| Batch size | 100 | 256 | Different |
| Epoch count | Follow the paper's training procedure and model-selection protocol | 300 epochs by default | Needs protocol alignment and documentation |
| Objective | Multiclass classification | `CrossEntropyLoss` on classifier logits and integer labels | Standard implementation choice; no softmax should be applied before this loss |
| Evaluation | Report the paper's task-specific metrics and protocol | Script logs accuracy, macro-F1, and weighted-F1 | Metrics overlap in spirit, but exact task/class handling and test protocol must match before numerical comparison |

## Critical correctness issue found

In `datasets.py`, the current implementation sets the class count using the number of target samples:

    self.n_classes = self.target.shape[0]

That value is the number of windows, not the number of activity classes. The training dataset had 43,985 windows, while the labels span 18 classes. The classifier must output one logit per class, not one logit per training example. A checkpoint created with the wrong output dimension must not be used as a valid final model.

Before training again:

1. Verify the label mapping and unique labels in train, validation, and test data.
2. Define `n_classes` from the intended class mapping (or the number of distinct labels after verifying that labels are contiguous and consistently encoded).
3. Confirm the classifier output shape is `(batch_size, 18)` for the current 18-class setup.
4. Re-run training and evaluation after fixing this issue. Treat previous metrics as diagnostic only, not final reproduction results.

## Recommended reproduction strategy

Maintain two clearly named configurations rather than mixing settings silently:

### A. Modern-code baseline

Use the modern repository's preprocessing, splits, window length, optimizer, and batch size after fixing the class-count bug. This establishes a valid baseline for this codebase, but it is not an exact reproduction of the paper.

### B. Paper-aligned experiment

Implement and document the paper's preprocessing/normalization, windowing, train/validation/test protocol, optimizer, batch size, and learning-rate behavior. Verify each setting against the paper and its original authors' implementation before claiming fidelity.

For both runs, record the code revision, random seeds, label mapping, split file lists, configuration, selected checkpoint criterion, and per-class metrics. Compare numerical results only when the task definition and evaluation protocol are compatible.

## Immediate next actions

- [ ] Correct and test the class-count definition before any new training run.
- [ ] Save a small configuration manifest describing the current modern baseline.
- [ ] Decide whether the project target is an exact paper-aligned reproduction or a corrected reproduction using the modern repository.
- [ ] Implement and evaluate the selected configuration; then compare test metrics with the corresponding paper results only if the task and protocol match.
