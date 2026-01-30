# CResUNet

We propose a coordination attention residual U-Net(CResU-Net) model designed to better capture the dynamic spatiotemporal correlations of high-resolution SST. CResU-Net is a deep learning model that integrates coordinate attention mechanisms, multiple residual modules, and depthwise separable convolutions.

## File Structure

config.py: Contains configuration settings.

run.py: Main training script (train/val/test).

train_utils.py: Training utilities (splits, losses, helpers).

inference.py: Inference helpers and visualization/metrics.

visualize.py: Training-time visualization utilities.

dataset.py: Dataset loading and preprocessing for NetCDF.

models/baseline/CResU_Net.py: Implementation of the CResU-Net model.

data/: NetCDF inputs.

results/: Saved models and outputs.
