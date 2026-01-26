import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import random

def visualize_prediction(model, val_loader, device, epoch, save_dir='./results'):
    """
    可视化订正结果
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model.eval()
    
    # --- 1. 抽取样本 ---
    val_dataset = val_loader.dataset
    total_samples = len(val_dataset)
    random_idx = random.randint(0, total_samples - 1)
    
    print(f"------------ 可视化抽样调试 ------------")
    print(f"Epoch {epoch}: 正在从 {total_samples} 个验证集样本中抽取第 [{random_idx}] 号样本...")
    
    x, y, mask = val_dataset[random_idx]
    
    # 增加 Batch 维度
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 模型输出 Bias
        pred_bias = model(x)
    
    # --- 2. 还原 SST ---
    input_sst_tensor = x[0, :120]
    pred_bias_tensor = pred_bias[0]
    
    # Corrected = Forecast - Bias
    corrected_sst_tensor = input_sst_tensor - pred_bias_tensor
    
    input_sst = input_sst_tensor.cpu().numpy()
    target_sst = y[0].cpu().numpy()
    pred_sst = corrected_sst_tensor.cpu().numpy()
    valid_mask = mask[0].cpu().numpy()
    
    # --- 3. 有效性判断 ---
    valid_steps_flag = valid_mask.max(axis=(1, 2))
    valid_indices = np.where(valid_steps_flag == 1)[0]
    
    if len(valid_indices) == 0:
        print(f"❌ 抽到了无效样本，跳过。")
        return

    last_step = valid_indices[-1]
    
    # 增加中间点
    if last_step > 24:
        mid_step = (24 + last_step) // 2
        raw_steps = [0, 24, mid_step, last_step]
    else:
        raw_steps = [0, 24, last_step]

    plot_steps = sorted(list(set(raw_steps)))
    plot_steps = [t for t in plot_steps if t <= last_step]

    # --- 4. 绘图 ---
    # 陆地掩码处理
    input_sst[valid_mask == 0] = np.nan
    target_sst[valid_mask == 0] = np.nan
    pred_sst[valid_mask == 0] = np.nan
    
    for t in plot_steps:
        if np.isnan(target_sst[t]).all(): continue
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 计算 Bias
        original_bias = input_sst[t] - target_sst[t]
        corrected_bias = pred_sst[t] - target_sst[t]
        
        # --- 【关键修改】计算 RMSE (先平方均值，再开根号) ---
        orig_rmse = np.sqrt(np.nanmean(original_bias**2))
        corr_rmse = np.sqrt(np.nanmean(corrected_bias**2))
        
        try:
            vmin = min(np.nanmin(input_sst[t]), np.nanmin(target_sst[t]))
            vmax = max(np.nanmax(input_sst[t]), np.nanmax(target_sst[t]))
        except ValueError: continue

        cmap = plt.cm.jet
        cmap.set_bad(color='white')
        
        # Row 1: 场图
        im1 = axes[0, 0].imshow(input_sst[t], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 0].set_title(f"Forecast T={t}h")
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(target_sst[t], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 1].set_title(f"Ground Truth")
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(pred_sst[t], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 2].set_title(f"Corrected")
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Row 2: 误差图
        bias_cmap = plt.cm.seismic
        bias_cmap.set_bad(color='white')
        
        curr_bias_max = max(np.nanmax(np.abs(original_bias)), np.nanmax(np.abs(corrected_bias)))
        if curr_bias_max == 0: curr_bias_max = 0.1
        
        # 显示 RMSE
        im4 = axes[1, 0].imshow(original_bias, cmap=bias_cmap, vmin=-curr_bias_max, vmax=curr_bias_max, origin='lower')
        axes[1, 0].set_title(f"Original Bias\nRMSE: {orig_rmse:.4f}°C") # 加上单位
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 2].imshow(corrected_bias, cmap=bias_cmap, vmin=-curr_bias_max, vmax=curr_bias_max, origin='lower')
        axes[1, 2].set_title(f"Residual Bias\nRMSE: {corr_rmse:.4f}°C") # 加上单位
        plt.colorbar(im5, ax=axes[1, 2])
        
        # 中间文字说明
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 
                        f"Improvement:\nFrom {orig_rmse:.4f}\nTo {corr_rmse:.4f}", 
                        ha='center', va='center', fontsize=16, fontweight='bold')
        
        plt.suptitle(f"SST Correction (Epoch {epoch}) @ T={t}h\nSample Index: {random_idx}", fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'vis_epoch{epoch}_T{t}.png')
        plt.savefig(save_path)
        plt.close()