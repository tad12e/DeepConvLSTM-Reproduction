# Reproduction Status

## Current status

The repository currently contains a cleaned Colab notebook documenting the OPPORTUNITY workflow. It clones and uses the upstream implementation at https://github.com/STRCWearlab/DeepConvLSTM_py3. The uploaded bundle did not include the upstream Python source files, so those files are not represented as locally maintained code in this repository yet.

## Known issue from the earlier run

During checkpoint inspection, the classifier output layer was found to have 43,985 outputs, matching the number of training windows rather than the 18 activity labels. The dataset class used a definition equivalent to:

    self.n_classes = self.target.shape[0]

This counts target samples, not distinct classes. The class-count logic needs to be corrected and verified before a new training run. Check the actual label mapping and number of classes rather than relying on the number of windows.

## Interpretation of previous metrics

Metrics produced by the run with the incorrect classifier output size are not valid final reproduction results. Do not cite them as the final result. Re-run training and evaluation only after fixing the class-count issue.

## Paper fidelity

This project uses a modern PyTorch implementation as a starting point. It should not be described as an exact reproduction until the architecture, preprocessing and normalization, participant/session split, windowing, optimizer, learning-rate schedule, batch size, and evaluation protocol have been checked against the paper and documented.

## Data and artifacts

Do not commit the OPPORTUNITY dataset, generated .npz files, or model checkpoints. Store these separately (for example, in Google Drive) and document how to obtain or generate them.
