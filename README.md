# CResUNet

We propose a coordination attention residual U-Net(CResU-Net) model designed to better capture the dynamic spatiotemporal correlations of high-resolution SST. CResU-Net is a deep learning model that integrates coordinate attention mechanisms, multiple residual modules, and depthwise separable convolutions.

## File Structure

config.py: Contains configuration settings.

train.py: Main training script (train/val/test).

train_macom.py: MaCOM 新管线训练入口（高分辨率输入 + GLO12 低分辨率监督）。

train_utils.py: Training utilities (splits, losses, helpers).

predict_utils.py: Inference helpers and visualization/metrics.

predict.py: CLI entry for full-file/time-range correction and structured NetCDF output.

data/data_process_utils.py: Data processing helpers for raw-NC interpolation and structured NetCDF writing.

data/data_process.py: CLI entry for converting raw forecast/reanalysis files into structured NetCDF.

visualize.py: Training-time visualization utilities.

dataset.py: Dataset loading and preprocessing for NetCDF.

dataset_macom.py: MaCOM 全图数据集（完整 1168×1000 网格）。

downsample.py: MaCOM ↔ GLO12 的下采样/上采样与 grid 构建。

models/baseline/CResU_Net.py: Implementation of the CResU-Net model.

data/: NetCDF inputs.

data/glo12_reader.py: GLO12 多文件读取与时间对齐。

results/: Saved models and outputs.

run_predict.sh: Shell wrapper for batch inference (conda env, date-range filtering, optional bias output)。

run_data_process.sh: Shell wrapper for raw data processing (date-range filtering, optional reanalysis output).

run_all.sh: End-to-end pipeline wrapper (data processing + prediction)。

## 流程&模型简介

### MaCOM 新管线

MaCOM 管线利用深度学习模型对高分辨率海洋预报 SST 进行智能订正，使其更接近 GLO12 再分析产品精度。

**数据流**：高分辨率 MaCOM 预报（1168×1000 网格，168 小时预报时效）作为输入，低分辨率 GLO12 再分析（25×26 网格）作为订正目标。模型学习预报与再分析之间的系统性偏差，并在全图尺度上一次完成订正。

**模型架构**：基于 CResU-Net（协同注意力残差 U-Net），输入 171 通道（168 小时 SST 预报 + 陆地掩膜 + 经纬度位置编码），输出 168 通道偏差场。模型在全图尺度上训练与推理，天然具备全局空间上下文感知能力。

**训练**：运行 `train_macom.py`，结果输出至 `train_results_macom/`。损失函数以 RMSE 为主，辅以 TV 平滑约束和 L1 背景抑制，所有计算均在 GLO12 低分辨率网格上通过海洋掩膜归一化完成，有效避免陆地像素污染。

**推理与评估**：使用 `predict_demo_macom.ipynb` 进行单文件全图推理、高/低分辨率对比可视化和全年误差统计。

### 旧管线（macom_compat / fvcom）

1. 使用`data/interpolate.ipynb`将原始FVCOM网格数据插值到约0.01˚x0.01˚的Grid网格。插值过程基于FVCOM局部格点密度智能生成陆地mask。

2. 运行`train.py`进行训练，输出结果位于`train_results`目录下，包含最佳模型参数、loss曲线、以及每次性能提升后部分时刻的展示。注意每次运行前会清空`train_results`目录，因此获得满意的训练结果后建议首先复制备份。

- 模型的输入有121通道，其中120通道为数值模式对未来120小时的hourly预测结果，另外一通道为陆地mask。监督数据则来自全年的hourly再分析数据，最终输出120通道每日经过智能订正的未来120小时hourly预测结果。原始与监督数据的差异较小，因此实际学习与输出的为需订正的残差。默认在每个月底划分5天测试集与5天验证集，其余日期数据作为训练集。预测数据与再分析数据时间范围一致（1月1日至12月31日），而12月底的预测数据可能超出再分析数据范围，训练时能够智能裁切仅使用存在再分析数据的部分。

- 损失函数中最主要的是`weighted_masked_rmse_loss`，以偏差值的RMSE为基础。在训练过程中发现前25小时的预测数据和再分析数据似乎完全一致，因此计算损失函数时屏蔽前25小时；此外误差总体随时间增加误差逐渐增大，因此计算时赋予的权重也随时间增加而增加。`total_variation_loss`添加了TV Loss，并且打开CResUNet的bilinear模式，平滑模型输出避免出现“棋盘格”效应。`smart_background_l1_loss`添加L1 Loss，抑制模型对接近零偏差区域的无谓修正，避免为原本无需订正的区域引入新的误差。

- 文件路径、数据划分和模型超参数保存在`config.py`中，可以按需修改后再运行模型。

3. `run_all.sh`用于一键执行“数据处理 + 订正推理”全流程：

- 阶段1（`run_data_process.sh`）：调用`data/data_process.py`，将原始`sst_forecast/*.nc`插值整合为`data/forecast_structured.nc`（默认），并可选生成`data/reanalysis_structured.nc`。
- 阶段2（`run_predict.sh`）：调用`predict.py`，读取`data/forecast_structured.nc`并输出`out/forecast_corrected_structured.nc`。
- 可通过`START_DATE`和`END_DATE`同时控制两个阶段的日期范围（例如按月分批）。
- `SAVE_REANALYSIS=0`为默认设置（仅生成predict所需的`forecast_structured.nc`）；设为`1`时额外生成`reanalysis_structured.nc`。
- `SAVE_BIAS=0`为默认设置（仅保存订正后的`sst`）；设为`1`时预测结果额外保存`pred_bias`变量。
- 支持`RUN_DATA_PROCESS`和`RUN_PREDICT`阶段开关，可单独执行任一阶段。

4. `predict_demo.ipynb`用于推理结果检查与调试，当前分为两部分：

- **订正脚本运行结果加载**：读取`run_predict.sh`/`predict.py`生成的结构化输出`out/forecast_corrected_structured.nc`，按`run_idx + t`展示Forecast、Corrected与二者间的Bias。

- **SST 订正推理调试**：直接加载`train_results/best_model.pth`中的训练结果，对照监督数据进行单日可视化分析，并评估全年预测误差，便于对模型行为做快速诊断。