import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import NCCorrectionDataset  # 导入新写的 Dataset
from models.baseline.CResU_Net import CRUNet
from config import experiment_params, data_params, model_params
from visualize import visualize_prediction
import os
import shutil
import random

def masked_mse_loss(pred, target, mask):
    """
    只计算有效区域(mask=1)的均方误差
    """
    diff = (pred - target) ** 2
    # 只保留有效部分的误差
    valid_diff = diff * mask 
    
    # 计算平均值：总误差 / 有效像素数
    # 加上 1e-6 防止除以 0
    loss = valid_diff.sum() / (mask.sum() + 1e-6)
    return loss

def total_variation_loss(img, weight=1.0):
    """
    计算全变分损失 (Total Variation Loss)，用于平滑图像
    img: [Batch, Time, H, W]
    """
    b, t, h, w = img.size()
    
    # 计算水平方向差异 (右边像素 - 左边像素)
    tv_h = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    
    # 计算垂直方向差异 (下边像素 - 上边像素)
    tv_w = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    
    return weight * (tv_h + tv_w) / (b * t * h * w)

def weighted_masked_rmse_loss(pred, target, mask, start_w=1.0, end_w=5.0, epsilon=1e-6):
    """
    计算带时间权重的 RMSE Loss (Root Mean Squared Error)
    参数:
        start_w: T=0 时刻的权重 (默认1.0)
        end_w:   T=End 时刻的权重 (默认5.0)
    """
    # 1. 基础差异平方
    diff = (pred - target) ** 2
    
    # 2. 构造时间权重
    steps = pred.shape[1]
    weights = torch.linspace(start_w, end_w, steps, device=pred.device)
    weights = weights.view(1, -1, 1, 1)
    
    # 3. 应用权重和掩码
    weighted_diff = diff * weights * mask
    
    # 4. 计算 MSE
    mse = weighted_diff.sum() / (mask.sum() + 1e-6)
    
    # 5. 【关键修改】 开根号变成 RMSE
    # 加 epsilon 是为了防止 mse 为 0 时导致梯度为 NaN
    loss = torch.sqrt(mse + epsilon)
    
    return loss

def clear_output_dir(save_dir='./results'):
    """
    程序启动时清空输出目录，防止旧图片堆积
    """
    if os.path.exists(save_dir):
        # 递归删除整个文件夹
        shutil.rmtree(save_dir)
        print(f"已清空旧输出目录: {save_dir}")
    
    # 重新创建空文件夹
    os.makedirs(save_dir)

def run():
    # 1. 读取配置参数
    exp_cfg = experiment_params
    data_cfg = data_params
    model_cfg = model_params['CResU_Net']
    trainer_cfg = model_cfg['trainer']
    
    device = torch.device(exp_cfg['device'] if torch.cuda.is_available() else 'cpu')
    
    # 2. 环境初始化
    clear_output_dir(exp_cfg['save_dir'])
    
    # 3. 数据集初始化 (关键修改：不再使用 random_split)
    print("正在初始化训练集 ...")
    full_dataset = NCCorrectionDataset(
        data_cfg['forecast_path'], 
        data_cfg['reanalysis_path'], 
    )
    
    # 4. 手动划分索引 (前24个Run训练，后7个Run验证)
    total_runs = len(full_dataset)
    split_point = 24 # 或者使用 data_cfg['split']['train_run_count']
    
    indices = list(range(total_runs))
    # 设定种子，保证虽然是乱序，但每次乱得都一样 (方便复现)
    random.seed(42) 
    random.shuffle(indices)

    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    
    print(f"数据集划分完成 -> 训练集: {len(train_ds)}, 验证集: {len(val_ds)}")

    # 5. DataLoader
    batch_size = model_cfg['batch_gen']['batch_size']
    num_workers = data_cfg['num_workers']
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # 6. 模型初始化
    in_chan = model_cfg['core']['in_channels']
    out_chan = model_cfg['core']['out_channels']
    
    model = CRUNet(
        in_channels=in_chan, 
        out_channels=out_chan, 
        selected_dim=0,
        device=device
    ).to(device)
    
    # 7. 优化器 & 调度器 & 早停
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=trainer_cfg['learning_rate'],
        weight_decay=trainer_cfg['weight_decay']
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=trainer_cfg['lr_scheduler']['mode'],
        factor=trainer_cfg['lr_scheduler']['factor'],
        patience=trainer_cfg['lr_scheduler']['patience'],
    )
    
    # 早停控制变量
    best_val_loss = float('inf')
    early_stop_patience = trainer_cfg['early_stopping']['tolerance']
    early_stop_counter = 0
    
    # Loss 权重配置
    loss_w_start = trainer_cfg['loss_weights']['start_weight']
    loss_w_end = trainer_cfg['loss_weights']['end_weight']
    l1_w = trainer_cfg['loss_weights'].get('l1_weight', 0.0) # 默认0.0防报错

    loss_tv_weight = trainer_cfg['loss_weights'].get('tv_weight', 0.0) # 默认0.0防报错

    print(f"🚀 开始训练 (Epochs={trainer_cfg['num_epochs']})...")

    # 8. 训练循环
    for epoch in range(1, trainer_cfg['num_epochs'] + 1):
        model.train()
        train_combined_loss = 0
        train_rmse_total = 0
        train_tv_total = 0
        train_l1_total = 0
        
        
        # --- Training Step ---
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            # (1) 模型预测 Bias
            pred_bias = model(x)
            
            # (2) 计算 GT Bias (Input - Target)
            # 假设 input 在 x 的前120通道
            gt_bias = x[:, :120] - y 
            
            # (3) 计算 Loss
            # 1. 主 Loss (RMSE) - 负责准确性
            rmse_loss = weighted_masked_rmse_loss(pred_bias, gt_bias, mask, 
                                            start_w=loss_w_start, end_w=loss_w_end)
            # 2. 辅助 Loss (TV) - 负责平滑性
            tv_loss = total_variation_loss(pred_bias, weight=loss_tv_weight)

            # 3. L1 (稀疏 - 保护 0 值)
            l1_loss = (torch.abs(pred_bias) * mask).sum() / mask.sum()
            l1_loss = l1_loss * l1_w # 乘上权重

            # 3. 总 Loss
            loss = rmse_loss + tv_loss + l1_loss

            loss.backward()
            optimizer.step()

            train_combined_loss += loss.item()
            train_rmse_total += rmse_loss.item()
            train_tv_total += tv_loss.item()
            train_l1_total += l1_loss.item()
        
        steps = len(train_loader)
        avg_train_loss = train_combined_loss / steps
        avg_train_rmse = train_rmse_total / steps
        avg_train_tv = train_tv_total / steps
        avg_train_l1 = train_l1_total / steps

        # --- Validation Step ---
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                
                pred_bias = model(x)
                gt_bias = x[:, :120] - y
                loss = weighted_masked_rmse_loss(pred_bias, gt_bias, mask,
                                                start_w=loss_w_start, end_w=loss_w_end)
                val_loss_total += loss.item()
        
        avg_val_loss = val_loss_total / len(val_loader)

        # --- 日志与调度 ---
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{trainer_cfg['num_epochs']} | "
              f"Val RMSE: {avg_val_loss:.6f} | "
              f"Train RMSE: {avg_train_rmse:.4f} "
              f"(TV: {avg_train_tv:.4f}, L1: {avg_train_l1:.4f}) | "
              f"LR: {current_lr:.2e}")
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # --- 早停与保存 ---
        # 加上微小的 delta 阈值，防止 Loss 震荡导致误判
        delta = trainer_cfg['early_stopping']['delta']
        
        if avg_val_loss < (best_val_loss - delta):
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            # 保存模型
            save_path = os.path.join(exp_cfg['save_dir'], exp_cfg['model_save_name'])
            torch.save(model.state_dict(), save_path)
            print(f"--> ✨ 性能提升！模型已保存: {save_path}")
        else:
            early_stop_counter += 1
            print(f"--> 💤 性能未提升 ({early_stop_counter}/{early_stop_patience})")
            
        if early_stop_counter >= early_stop_patience:
            print(f"🛑 触发早停机制，训练结束。")
            break

        # --- 可视化 ---
        visualize_prediction(model, val_loader, device, epoch, save_dir=exp_cfg['save_dir'])
    
    print(f"🏁 训练流程结束。最佳验证集 Loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    run()