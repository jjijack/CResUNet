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
    
    # --- 1. 直接获取 Dataset 对象抽取样本 ---
    val_dataset = val_loader.dataset
    total_samples = len(val_dataset)
    
    random_idx = random.randint(0, total_samples - 1)
    
    print(f"------------ 可视化抽样调试 ------------")
    print(f"Epoch {epoch}: 正在从 {total_samples} 个验证集样本中抽取第 [{random_idx}] 号样本...")
    
    # 获取数据 [C, H, W]
    x, y, mask = val_dataset[random_idx]
    
    # 增加 Batch 维度: [1, C, H, W]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(x)
    
    # --- 2. 提取数据 ---
    input_sst = x[0, :120].cpu().numpy()
    target_sst = y[0].cpu().numpy()
    pred_sst = pred[0].cpu().numpy()
    valid_mask = mask[0].cpu().numpy()
    
    # --- 3. 有效性判断 ---
    valid_steps_flag = valid_mask.max(axis=(1, 2))
    valid_indices = np.where(valid_steps_flag == 1)[0]
    
    if len(valid_indices) == 0:
        print(f"❌ 抽到了无效样本 (Mask全0)，跳过。")
        return

    last_step = valid_indices[-1]
    print(f"✅ 样本有效时长: {last_step + 1} 小时")

    plot_steps = sorted(list(set([0, 24, last_step])))
    plot_steps = [t for t in plot_steps if t <= last_step]

    # --- 4. 绘图 ---
    # 全局 NaN 处理 (陆地变白)
    input_sst[valid_mask == 0] = np.nan
    target_sst[valid_mask == 0] = np.nan
    pred_sst[valid_mask == 0] = np.nan
    
    for t in plot_steps:
        if np.isnan(target_sst[t]).all(): continue
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 计算 Bias
        original_bias = input_sst[t] - target_sst[t]
        corrected_bias = pred_sst[t] - target_sst[t]
        
        # 计算 MSE (使用 nanmean 忽略陆地)
        orig_mse = np.nanmean(original_bias**2)
        corr_mse = np.nanmean(corrected_bias**2)
        
        try:
            vmin = min(np.nanmin(input_sst[t]), np.nanmin(target_sst[t]))
            vmax = max(np.nanmax(input_sst[t]), np.nanmax(target_sst[t]))
        except ValueError: continue

        cmap = plt.cm.jet
        cmap.set_bad(color='white')
        
        # Row 1: 场图
        im1 = axes[0, 0].imshow(input_sst[t], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        title_suffix = " (Last Hour)" if t == last_step else ""
        axes[0, 0].set_title(f"Forecast T={t}h{title_suffix}")
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
        
        im4 = axes[1, 0].imshow(original_bias, cmap=bias_cmap, vmin=-curr_bias_max, vmax=curr_bias_max, origin='lower')
        axes[1, 0].set_title(f"Original Bias\nMSE: {orig_mse:.4f}")
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 2].imshow(corrected_bias, cmap=bias_cmap, vmin=-curr_bias_max, vmax=curr_bias_max, origin='lower')
        axes[1, 2].set_title(f"Residual Bias\nMSE: {corr_mse:.4f}")
        plt.colorbar(im5, ax=axes[1, 2])
        
        # --- 中间的文字说明 ---
        axes[1, 1].axis('off') # 关掉坐标轴
        axes[1, 1].text(0.5, 0.5, 
                        f"Improvement:\nFrom {orig_mse:.4f}\nTo {corr_mse:.4f}", 
                        ha='center', va='center', fontsize=16, fontweight='bold')
        
        plt.suptitle(f"SST Correction (Epoch {epoch}) @ T={t}h\nSample Index: {random_idx}", fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'vis_epoch{epoch}_T{t}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"图片已保存: {save_path}")