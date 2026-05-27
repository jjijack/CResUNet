# CResUNet

We propose a coordination attention residual U-Net(CResU-Net) model designed to better capture the dynamic spatiotemporal correlations of high-resolution SST. CResU-Net is a deep learning model that integrates coordinate attention mechanisms, multiple residual modules, and depthwise separable convolutions.

## File Structure

config.py: Contains configuration settings.

train_macom.py: MaCOM 新管线训练入口（高分辨率输入 + GLO12 低分辨率监督）。

train_fvcom.py: FVCOM 旧管线训练入口（规则网格）。

train_utils.py: Training utilities (splits, losses, helpers).

predict_utils.py: Inference helpers and visualization/metrics.

predict_macom.py: MaCOM 新管线推理入口（全图推理，输出高分辨率订正 SST）。

predict_fvcom.py: FVCOM 旧管线推理入口（结构化 NetCDF 输出）。

data/data_process_utils.py: Data processing helpers for raw-NC interpolation and structured NetCDF writing.

data/data_process_fvcom.py: CLI entry for converting raw FVCOM forecast/reanalysis files into structured NetCDF.

visualize.py: Training-time visualization utilities.

dataset.py: Dataset loading and preprocessing for NetCDF.

dataset_macom.py: MaCOM 全图数据集（完整 1168×1000 网格）。

downsample.py: MaCOM ↔ GLO12 的下采样/上采样与 grid 构建。

models/baseline/CResU_Net.py: Implementation of the CResU-Net model.

data/: NetCDF inputs.

data/glo12_reader.py: GLO12 多文件读取与时间对齐。

train_results_macom/: MaCOM 管线训练结果。

train_results/: FVCOM 管线训练结果。

out_macom/: MaCOM 管线推理输出。

out/: FVCOM 管线推理输出。

run_predict.sh: Shell wrapper for batch inference (macom / fvcom 通用)。

run_data_process.sh: Shell wrapper for raw data processing (fvcom 专用)。

run_all.sh: End-to-end pipeline wrapper (macom / fvcom 通用)。

predict_demo_macom.py: MaCOM 订正结果可视化 CLI（读 out_macom/ 绘对比图）。

predict_demo_fvcom.py: FVCOM 订正结果可视化 CLI（读 out/ 绘对比图）。

## 流程&模型简介

### MaCOM 新管线

MaCOM 管线利用深度学习模型对高分辨率海洋预报 SST 进行智能订正，使其更接近 GLO12 再分析产品精度。

**数据流**：高分辨率 MaCOM 预报（1168×1000 网格，168 小时预报时效）作为输入，低分辨率 GLO12 再分析（25×26 网格）作为订正目标。模型学习预报与再分析之间的系统性偏差，并在全图尺度上一次完成订正。

**模型架构**：基于 CResU-Net（协同注意力残差 U-Net），输入 171 通道（168 小时 SST 预报 + 陆地掩膜 + 经纬度位置编码），输出 168 通道偏差场。模型在全图尺度上训练与推理，天然具备全局空间上下文感知能力。

**训练**：运行 `train_macom.py`，结果输出至 `train_results_macom/`。损失函数以 RMSE 为主，辅以 TV 平滑约束和 L1 背景抑制，所有计算均在 GLO12 低分辨率网格上通过海洋掩膜归一化完成，有效避免陆地像素污染。

**推理与评估**：使用 `predict_demo_macom.ipynb`（交互式笔记本）或 `predict_demo_macom.py --date YYYYMMDD --step H`（CLI）进行单文件全图推理和对比可视化。

### 旧管线（fvcom）

1. 使用`data/interpolate.ipynb`将原始FVCOM网格数据插值到约0.01˚x0.01˚的Grid网格。插值过程基于FVCOM局部格点密度智能生成陆地mask。

2. 运行`train_fvcom.py`进行训练，输出结果位于`train_results`目录下，包含最佳模型参数、loss曲线、以及每次性能提升后部分时刻的展示。注意每次运行前会清空`train_results`目录，因此获得满意的训练结果后建议首先复制备份。

- 模型的输入有121通道，其中120通道为数值模式对未来120小时的hourly预测结果，另外一通道为陆地mask。监督数据则来自全年的hourly再分析数据，最终输出120通道每日经过智能订正的未来120小时hourly预测结果。原始与监督数据的差异较小，因此实际学习与输出的为需订正的残差。默认在每个月底划分5天测试集与5天验证集，其余日期数据作为训练集。预测数据与再分析数据时间范围一致（1月1日至12月31日），而12月底的预测数据可能超出再分析数据范围，训练时能够智能裁切仅使用存在再分析数据的部分。

- 损失函数中最主要的是`weighted_masked_rmse_loss`，以偏差值的RMSE为基础。在训练过程中发现前25小时的预测数据和再分析数据似乎完全一致，因此计算损失函数时屏蔽前25小时；此外误差总体随时间增加误差逐渐增大，因此计算时赋予的权重也随时间增加而增加。`total_variation_loss`添加了TV Loss，并且打开CResUNet的bilinear模式，平滑模型输出避免出现“棋盘格”效应。`smart_background_l1_loss`添加L1 Loss，抑制模型对接近零偏差区域的无谓修正，避免为原本无需订正的区域引入新的误差。

- 超参数在`train_fvcom.py`内部硬编码，独立于`config.py`的 MaCOM 设定。

3. `run_all.sh`支持两种管线切换（默认 macom，fvcom 需取消注释对应配置）：

- **macom 模式**：直接调`run_predict.sh` → `predict_macom.py`，读原始 MaCOM 文件输出高分辨率订正 SST 到`out_macom/`。
- **fvcom 模式**：`run_data_process.sh`调`data/data_process_fvcom.py`生成结构化网格；`run_predict.sh`调`predict_fvcom.py`输出到`out/`。
- `SAVE_BIAS=1`时输出`predicted_bias`而非`corrected_sst`（两管线均支持）。

4. 可视化：
- `predict_demo_fvcom.py`：读取`out/forecast_corrected_structured.nc`并绘图。
- `predict_demo_macom.py`：读取`out_macom/*_corrected.nc`，有 GLO12 时含对比，无 GLO12 时绘 Forecast / Corrected / Bias。