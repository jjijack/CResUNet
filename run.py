import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import NCCorrectionDataset
from models.baseline.CResU_Net import CRUNet
from config import experiment_params, data_params, model_params
from visualize import visualize_prediction
from train_utils import (
    clear_output_dir,
    create_monthly_split,
    total_variation_loss,
    weighted_masked_rmse_loss,
    smart_background_l1_loss,
)
import os
"""Training entry point."""

def run():
    # --- 配置读取 ---
    exp_cfg = experiment_params
    data_cfg = data_params
    model_cfg = model_params['CResU_Net']
    trainer_cfg = model_cfg['trainer']
    
    device = torch.device(exp_cfg['device'] if torch.cuda.is_available() else 'cpu')
    clear_output_dir(exp_cfg['save_dir'])

    # --- 数据集初始化 ---
    print("正在加载全量数据集 ...")
    full_dataset = NCCorrectionDataset(
        data_cfg['forecast_path'], 
        data_cfg['reanalysis_path'], 
    )
    
    run_dates = full_dataset.get_run_dates()

    # --- [关键] 按自然月切分 ---
    split_cfg = data_cfg.get('monthly_split', {})
    train_days = split_cfg.get('train_days', 20)
    val_days = split_cfg.get('val_days', 5)
    test_days = split_cfg.get('test_days', 5)

    train_idx, val_idx, test_idx = create_monthly_split(
        run_dates,
        train_days=train_days,
        val_days=val_days,
        test_days=test_days
    )
    
    # 创建 Subset (只是索引映射，不耗内存)
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    test_ds = Subset(full_dataset, test_idx)
    
    print(f"数据集划分完成 (Periodic Split):")
    print(f"  Train: {len(train_ds)} (反向传播)")
    print(f"  Val  : {len(val_ds)} (模型优选 & Buffer)")
    print(f"  Test : {len(test_ds)} (最终评估)")

    # DataLoader
    batch_size = model_cfg['batch_gen']['batch_size']
    num_workers = data_cfg['num_workers']
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    # Test loader 在训练完后再用

    # --- 模型初始化 ---
    model = CRUNet(
        in_channels=model_cfg['core']['in_channels'],
        out_channels=model_cfg['core']['out_channels'],
        selected_dim=0,
        device=device
    ).to(device)

    print("模型初始化完成")

    # --- 优化器 ---
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=trainer_cfg['learning_rate'],
        weight_decay=trainer_cfg['weight_decay']
    )
    
    # 推荐使用 Cosine 以跳出局部最优，或者继续用 ReduceLROnPlateau
    scheduler_cfg = trainer_cfg['lr_scheduler']
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg['mode'],
        factor=scheduler_cfg['factor'],
        patience=scheduler_cfg['patience']
    )
    
    # Loss 权重
    w_cfg = trainer_cfg['loss_weights']
    print(f"Loss配置: StartW={w_cfg['start_weight']}, EndW={w_cfg['end_weight']}, "
          f"TV={w_cfg['tv_weight']}, L1={w_cfg['l1_weight']}")

    # --- 训练循环 ---
    best_val_loss = float('inf')
    best_epoch = None
    early_stop_cnt = 0
    save_path = os.path.join(exp_cfg['save_dir'], exp_cfg['model_save_name'])
    history_train_rmse = []
    history_val_rmse = []

    for epoch in range(1, trainer_cfg['num_epochs'] + 1):
        model.train()
        log_meters = {'loss': 0, 'rmse': 0, 'tv': 0, 'l1': 0}
        
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            
            # Forward
            pred_bias = model(x)
            gt_bias = x[:, :120] - y # Forecast - Reanalysis
            
            # Loss Calculation
            # 1. RMSE (Main)
            rmse_loss = weighted_masked_rmse_loss(
                pred_bias,
                gt_bias,
                mask,
                start_w=w_cfg['start_weight'],
                end_w=w_cfg['end_weight'],
                ignore_steps=w_cfg.get('ignore_steps', 0),
                ignore_weight=w_cfg.get('ignore_weight', 0.0),
            )
            
            # 2. TV (Smoothing)
            tv_loss = total_variation_loss(pred_bias, weight=w_cfg['tv_weight'])
            
            # 3. Smart L1 (Background Suppression)
            # 只有当真实偏差 < 0.1 时，才惩罚 L1
            l1_loss = smart_background_l1_loss(pred_bias, gt_bias, mask, zero_threshold=0.1)
            l1_loss = l1_loss * w_cfg['l1_weight']
            
            # Total
            loss = rmse_loss + tv_loss + l1_loss
            
            loss.backward()
            optimizer.step()
            
            # Record
            log_meters['loss'] += loss.item()
            log_meters['rmse'] += rmse_loss.item()
            log_meters['tv'] += tv_loss.item()
            log_meters['l1'] += l1_loss.item()
            
        # Averages
        steps = len(train_loader)
        avg_train = {k: v/steps for k, v in log_meters.items()}

        # --- Validation ---
        model.eval()
        val_rmse_total = 0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                pred_bias = model(x)
                gt_bias = x[:, :120] - y
                
                # 验证集只看 RMSE
                v_loss = weighted_masked_rmse_loss(
                    pred_bias,
                    gt_bias,
                    mask,
                    start_w=w_cfg['start_weight'],
                    end_w=w_cfg['end_weight'],
                    ignore_steps=w_cfg.get('ignore_steps', 0),
                    ignore_weight=w_cfg.get('ignore_weight', 0.0),
                )
                val_rmse_total += v_loss.item()
        
        avg_val_rmse = val_rmse_total / len(val_loader)
        history_train_rmse.append(avg_train['rmse'])
        history_val_rmse.append(avg_val_rmse)

        # --- Log & Schedule ---
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{trainer_cfg['num_epochs']} | "
              f"Val RMSE: {avg_val_rmse:.5f} | "
              f"Train RMSE: {avg_train['rmse']:.4f} "
              f"(L1: {avg_train['l1']:.4f}, TV: {avg_train['tv']:.4f}) | "
              f"LR: {lr:.2e}")
        
        scheduler.step(avg_val_rmse)
        
        # --- Early Stopping & Save ---
        delta = trainer_cfg['early_stopping'].get('delta', 0.0)
        improved = avg_val_rmse < (best_val_loss - delta)
        if improved:
            best_val_loss = avg_val_rmse
            best_epoch = epoch
            early_stop_cnt = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> ✨ 性能提升！模型已保存: {save_path}")
        else:
            early_stop_cnt += 1
            print(f"--> 💤 性能未提升 ({early_stop_cnt}/{trainer_cfg['early_stopping']['tolerance']})")
            
        if early_stop_cnt >= trainer_cfg['early_stopping']['tolerance']:
            print(f"🛑 触发早停机制，训练结束。")
            break
            
        # 可视化 (仅在性能提升时)
        if improved:
            visualize_prediction(model, val_loader, device, epoch, save_dir=exp_cfg['save_dir'])

    # 训练曲线
    if history_train_rmse and history_val_rmse:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(history_train_rmse) + 1), history_train_rmse, label='Train RMSE')
        plt.plot(range(1, len(history_val_rmse) + 1), history_val_rmse, label='Val RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training Curve')
        plt.legend()
        plt.tight_layout()
        curve_path = os.path.join(exp_cfg['save_dir'], 'loss_curve.png')
        plt.savefig(curve_path)
        plt.close()
        print(f"训练曲线已保存: {curve_path}")

    if best_epoch is not None:
        print(f"最佳模型来自第 {best_epoch} 个 epoch，Val RMSE: {best_val_loss:.6f}")

    # ==========================================
    # 3. 最终测试 (Test Step)
    # ==========================================
    print("\n" + "="*40)
    print("🏆 训练结束，开始在 Test Set 上评估最佳模型...")
    print("="*40)
    
    # 加载最佳权重
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    test_rmse_total = 0
    test_l1_total = 0
    
    with torch.no_grad():
        for x, y, mask in test_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred_bias = model(x)
            gt_bias = x[:, :120] - y
            
            # 计算纯粹的 RMSE (不带时间权重，或者带权重看需求，这里计算带权重的以便对比)
            t_loss = weighted_masked_rmse_loss(
                pred_bias,
                gt_bias,
                mask,
                start_w=w_cfg['start_weight'],
                end_w=w_cfg['end_weight'],
                ignore_steps=w_cfg.get('ignore_steps', 0),
                ignore_weight=w_cfg.get('ignore_weight', 0.0),
            )
            
            # 计算平均绝对误差 (MAE) 看物理量级
            mae = (torch.abs(pred_bias - gt_bias) * mask).sum() / (mask.sum() + 1e-6)
            
            test_rmse_total += t_loss.item()
            test_l1_total += mae.item()
            
    avg_test_rmse = test_rmse_total / len(test_loader)
    avg_test_mae = test_l1_total / len(test_loader)
    
    print(f"📊 最终测试集结果 (Test Set):")
    print(f"   RMSE Score : {avg_test_rmse:.5f}")
    print(f"   MAE  Score : {avg_test_mae:.5f} °C")
    print(f"   模型保存路径: {save_path}")

if __name__ == '__main__':
    run()