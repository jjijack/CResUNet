# CResUNet

We propose a coordination attention residual U-Net(CResU-Net) model designed to better capture the dynamic spatiotemporal correlations of high-resolution SST. CResU-Net is a deep learning model that integrates coordinate attention mechanisms, multiple residual modules, and depthwise separable convolutions.

## File Structure

config.py: Contains configuration settings.

train.py: Main training script (train/val/test).

train_utils.py: Training utilities (splits, losses, helpers).

predict_utils.py: Inference helpers and visualization/metrics.

predict.py: CLI entry for full-file/time-range correction and structured NetCDF output.

visualize.py: Training-time visualization utilities.

dataset.py: Dataset loading and preprocessing for NetCDF.

models/baseline/CResU_Net.py: Implementation of the CResU-Net model.

data/: NetCDF inputs.

results/: Saved models and outputs.

run_predict.sh: Shell wrapper for batch inference (conda env, date-range filtering, optional bias output).

## 流程&模型简介

1. 使用`data/interpolate.ipynb`将原始FVCOM网格数据插值到约0.01˚x0.01˚的Grid网格。插值过程基于FVCOM局部格点密度智能生成陆地mask。

2. 运行`train.py`进行训练，输出结果位于`train_results`目录下，包含最佳模型参数、loss曲线、以及每次性能提升后部分时刻的展示。注意每次运行前会清空`train_results`目录，因此获得满意的训练结果后建议首先复制备份。

- 模型的输入有121通道，其中120通道为数值模式对未来120小时的hourly预测结果，另外一通道为陆地mask。监督数据则来自全年的hourly再分析数据，最终输出120通道每日经过智能订正的未来120小时hourly预测结果。原始与监督数据的差异较小，因此实际学习与输出的为需订正的残差。默认在每个月底划分5天测试集与5天验证集，其余日期数据作为训练集。预测数据与再分析数据时间范围一致（1月1日至12月31日），而12月底的预测数据可能超出再分析数据范围，训练时能够智能裁切仅使用存在再分析数据的部分。

- 损失函数中最主要的是`weighted_masked_rmse_loss`，以偏差值的RMSE为基础。在训练过程中发现前25小时的预测数据和再分析数据似乎完全一致，因此计算损失函数时屏蔽前25小时；此外误差总体随时间增加误差逐渐增大，因此计算时赋予的权重也随时间增加而增加。`total_variation_loss`添加了TV Loss，并且打开CResUNet的bilinear模式，平滑模型输出避免出现“棋盘格”效应。`smart_background_l1_loss`添加L1 Loss，抑制模型对接近零偏差区域的无谓修正，避免为原本无需订正的区域引入新的误差。

- 文件路径、数据划分和模型超参数保存在`config.py`中，可以按需修改后再运行模型。

3. `run_predict.sh`用于生产式运行整文件订正流程：

- 默认调用`predict.py`读取`data/forecast_structured.nc`，输出结构化文件`out/forecast_corrected_structured.nc`。
- 可通过`CONDA_ENV_NAME`指定运行环境；为空时使用当前Python环境。
- 可通过`START_DATE`和`END_DATE`只预测指定时间范围（例如按月分批）。
- `SAVE_BIAS=0`为默认设置，仅保存订正后的`sst`；若设为`1`则额外保存`pred_bias`变量。

4. `predict_demo.ipynb`用于推理结果检查与调试，当前分为两部分：

- **订正脚本运行结果加载**：读取`run_predict.sh`/`predict.py`生成的结构化输出`out/forecast_corrected_structured.nc`，按`run_idx + t`展示Forecast、Corrected与二者间的Bias。

- **SST 订正推理调试**：直接加载`train_results/best_model.pth`中的训练结果，对照监督数据进行单日可视化分析，并评估全年预测误差，便于对模型行为做快速诊断。