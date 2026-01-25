import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def visualize_prediction(model, val_loader, device, save_dir='./results'):
    """
    可视化订正结果：抽取一个Batch，画出空间分布图和误差图
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model.eval()
    
    # 1. 获取一个 Batch 的数据
    with torch.no_grad():
        # 注意：你的 Dataset 返回的是 (x, y, mask)
        x, y, mask = next(iter(val_loader))
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        
        # 2. 模型推理
        pred = model(x)
        
        # 3. 转回 CPU numpy (取 Batch 中的第 0 个样本)
        # 形状变成: [120, H, W]
        input_sst = x[0].cpu().numpy()
        target_sst = y[0].cpu().numpy()
        pred_sst = pred[0].cpu().numpy()
        valid_mask = mask[0].cpu().numpy()

    # --- 绘图配置 ---
    # 我们选择几个关键时间点来画图，比如第 1 小时, 第 24 小时, 第 120 小时
    time_steps = [0, 24, 119] 
    
    for t in time_steps:
        # 如果这个时间点没有真值 (Mask=0)，跳过
        if valid_mask[t].sum() == 0:
            print(f"时间点 T={t} 无有效真值，跳过绘图。")
            continue
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 计算误差
        # 原始误差 = 预报 - 真值
        original_bias = input_sst[t] - target_sst[t]
        # 订正后误差 = 订正值 - 真值
        corrected_bias = pred_sst[t] - target_sst[t]
        
        # 统一 Colorbar 范围 (SST范围)
        vmin = min(input_sst[t].min(), target_sst[t].min())
        vmax = max(input_sst[t].max(), target_sst[t].max())
        
        # --- 第一行：SST 场对比 ---
        
        # 1. 原始数值预报 (Forecast)
        im1 = axes[0, 0].imshow(input_sst[t], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 0].set_title(f"Original Forecast (T={t}h)")
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # 2. 观测真值 (Reanalysis)
        im2 = axes[0, 1].imshow(target_sst[t], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 1].set_title(f"Ground Truth (Reanalysis)")
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. CResU-Net 订正结果
        im3 = axes[0, 2].imshow(pred_sst[t], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 2].set_title(f"CResU-Net Corrected")
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # --- 第二行：误差 (Bias) 对比 ---
        # 误差通常在 0 附近，用 seismic 或 bwr 色标，0 为白色
        bias_max = max(abs(original_bias).max(), abs(corrected_bias).max())
        bias_min = -bias_max
        
        # 4. 原始偏差
        im4 = axes[1, 0].imshow(original_bias, cmap='seismic', vmin=bias_min, vmax=bias_max, origin='lower')
        axes[1, 0].set_title(f"Original Bias (Forecast - Truth)\nMSE: {np.mean(original_bias**2):.4f}")
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 5. 订正后偏差
        im5 = axes[1, 2].imshow(corrected_bias, cmap='seismic', vmin=bias_min, vmax=bias_max, origin='lower')
        axes[1, 2].set_title(f"Residual Bias (Corrected - Truth)\nMSE: {np.mean(corrected_bias**2):.4f}")
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # 中间的图留空或者画点别的
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, f"Improvement:\nFrom {np.mean(original_bias**2):.4f}\nTo {np.mean(corrected_bias**2):.4f}", 
                        ha='center', va='center', fontsize=15)

        plt.suptitle(f"SST Correction Result @ T={t} hours", fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'vis_T{t}.png')
        plt.savefig(save_path)
        print(f"图片已保存: {save_path}")
        plt.close()