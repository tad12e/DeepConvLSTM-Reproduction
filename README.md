# DeepConvLSTM Reproduction

This repository tracks an attempt to reproduce the DeepConvLSTM human activity recognition model on the OPPORTUNITY dataset.

## Current workflow

The Colab notebook documents the working process:
1. Clone the modern PyTorch implementation from [STRCWearlab/DeepConvLSTM_py3](https://github.com/STRCWearlab/DeepConvLSTM_py3).
2. Run the OPPORTUNITY preprocessing script.
3. inspect the generated train, validation, and test windows and labels.
4. Run the training script and save checkpoints to Google Drive.
5. Inspect the saved checkpoint and model parameter shapes.

## Important status note

This is a reproduction-in-progress, not yet a verified exact reproduction of the original paper. The modern implementation differs from the paper in preprocessing, data splits, and some training settings. A classifier output-size issue was also identified during inspection, so metrics from the earlier run should not be treated as valid final results until that issue is corrected and training is rerun.

## Data and checkpoints

The OPPORTUNITY dataset and model checkpoints are not committed to this repository. Keep datasets, generated results, and large checkpoint files in local storage or Google Drive.

## Current contents

- `DeepConvLSTM_Reproduction.ipynb` — Colab workflow and investigation notes.

The Python implementation files are not included in the uploaded source bundle yet; the notebook currently clones the upstream implementation when run.
