"""
MaCOM 全图训练脚本。
直接输入完整 1168×1000 网格，不做 patch 切分。
消除拼接痕迹，模型学习全局空间上下文。
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from netCDF4 import Dataset as NCDataset, num2date

from config import experiment_params, data_params, model_params
from models.baseline.CResU_Net import CRUNet
from dataset_macom import MaCOMPatchDataset, _extract_datetime_from_filename
from data.glo12_reader import GLO12Reader
from downsample import build_downsample_grid, downsample_to_glo12, downsample_mask_nearest
from train_utils import (
    clear_output_dir,
    total_variation_loss,
    weighted_masked_rmse_loss,
    smart_background_l1_loss,
    create_monthly_split,
)


def run():
    exp_cfg = experiment_params
    data_cfg = data_params
    macom_cfg = data_cfg["macom"]
    model_cfg = model_params["CResU_Net"]
    trainer_cfg = model_cfg["trainer"]

    device = torch.device(exp_cfg["device"] if torch.cuda.is_available() else "cpu")
    clear_output_dir(exp_cfg["save_dir"])

    glo12 = GLO12Reader(macom_cfg["glo12_pattern"], tolerance_hours=macom_cfg["time_tolerance_hours"])
    print(f"GLO12: {glo12.shape}")

    print("正在初始化全图数据集 ...")
    full_dataset = MaCOMPatchDataset(
        forecast_pattern=macom_cfg["forecast_pattern"],
        glo12_pattern=macom_cfg["glo12_pattern"],
        glo12_reader=glo12,
        target_steps=macom_cfg["target_steps"],
        time_tolerance_hours=macom_cfg["time_tolerance_hours"],
    )
    src_h = full_dataset.src_h
    src_w = full_dataset.src_w
    padded_w = full_dataset.padded_w

    file_dates = [_extract_datetime_from_filename(f).date() for f in full_dataset.files]
    split_cfg = data_cfg.get("monthly_split", {})
    train_files, val_files, test_files = create_monthly_split(
        file_dates,
        train_days=split_cfg.get("train_days", 25),
        val_days=split_cfg.get("val_days", 3),
        test_days=split_cfg.get("test_days", 2),
    )

    train_ds = torch.utils.data.Subset(full_dataset, train_files)
    val_ds = torch.utils.data.Subset(full_dataset, val_files)
    test_ds = torch.utils.data.Subset(full_dataset, test_files)

    print(f"数据集划分: {len(full_dataset)} files")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    batch_size = model_cfg["batch_gen"]["batch_size"]
    num_workers = data_cfg["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # 全图 downsample grid（所有样本共用）
    grid = build_downsample_grid(
        full_dataset.src_lat, full_dataset.src_lon, src_h, src_w,
        glo12.lat, glo12.lon, len(glo12.lat), len(glo12.lon),
        y_offset=0, x_offset=0, patch_h=src_h, patch_w=src_w,
    ).to(device)

    grid_mask = (
        (grid[0, :, :, 0] >= -1.0) & (grid[0, :, :, 0] <= 1.0) &
        (grid[0, :, :, 1] >= -1.0) & (grid[0, :, :, 1] <= 1.0)
    ).float()

    src_ocean_full = full_dataset.ocean_mask  # 已在 dataset.__init__ 中预计算并 pad
    src_ocean = src_ocean_full[:, :src_w]
    glo12_ocean = downsample_mask_nearest(
        src_ocean, full_dataset.src_lat, full_dataset.src_lon,
        glo12.lat, glo12.lon,
        y_offset=0, x_offset=0, patch_h=src_h, patch_w=src_w,
    )

    model = CRUNet(
        in_channels=model_cfg["core"]["in_channels"],
        out_channels=model_cfg["core"]["out_channels"],
        selected_dim=0, device=device,
        base_channels=model_cfg["core"].get("base_channels", 64),
        dropout=model_cfg["core"].get("dropout", 0.0),
    ).to(device)
    print(f"模型: in={model_cfg['core']['in_channels']}, out={model_cfg['core']['out_channels']}, "
          f"base_ch={model_cfg['core'].get('base_channels', 64)}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=trainer_cfg["learning_rate"],
        weight_decay=trainer_cfg["weight_decay"],
    )
    scheduler_cfg = trainer_cfg["lr_scheduler"]
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_cfg["T_0"], T_mult=scheduler_cfg["T_mult"],
        eta_min=scheduler_cfg["eta_min"],
    )

    w_cfg = trainer_cfg["loss_weights"]
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_loss = float("inf")
    best_epoch = None
    early_stop_cnt = 0
    save_path = os.path.join(exp_cfg["save_dir"], exp_cfg["model_save_name"])
    history_train_rmse = []
    history_val_rmse = []

    for epoch in range(1, trainer_cfg["num_epochs"] + 1):
        model.train()
        log_meters = {"loss": 0, "rmse": 0, "tv": 0, "l1": 0}
        valid_batches = 0

        for x, y_lowres, mask_lowres in train_loader:
            x, y_lowres, mask_lowres = x.to(device), y_lowres.to(device), mask_lowres.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_bias = model(x)
                if not torch.isfinite(pred_bias).all():
                    continue

                mask_hr_full = x[:, 168:169]
                mask_hr = mask_hr_full[:, :, :, :src_w]
                forecast_hr = x[:, :168, :, :src_w]
                pred_bias_hr = pred_bias[:, :, :, :src_w]
                corrected_hr = forecast_hr - pred_bias_hr
                mask_lowres = mask_lowres * grid_mask[None, :, :] * torch.from_numpy(glo12_ocean).float().to(device)[None, None, :, :]
                if mask_lowres.sum().item() < 1.0:
                    continue

                forecast_lowres = downsample_to_glo12(forecast_hr, grid, mask=mask_hr)
                corrected_lowres = downsample_to_glo12(corrected_hr, grid, mask=mask_hr)

            corrected_lowres = corrected_lowres.float()
            forecast_lowres = forecast_lowres.float()
            y_lowres = y_lowres.float()
            mask_lowres = mask_lowres.float()
            pred_bias_f32 = pred_bias.float()

            if torch.isnan(corrected_lowres).any() or torch.isnan(forecast_lowres).any():
                continue

            gt_bias_lowres = forecast_lowres - y_lowres
            pred_bias_lowres = forecast_lowres - corrected_lowres

            rmse_loss = weighted_masked_rmse_loss(
                corrected_lowres, y_lowres, mask_lowres,
                start_w=w_cfg["start_weight"], end_w=w_cfg["end_weight"],
                ignore_steps=w_cfg.get("ignore_steps", 0),
                ignore_weight=w_cfg.get("ignore_weight", 0.0),
            )
            tv_loss = total_variation_loss(pred_bias_f32, mask=mask_hr_full.float(), weight=w_cfg["tv_weight"])
            l1_loss = smart_background_l1_loss(pred_bias_lowres, gt_bias_lowres, mask_lowres, zero_threshold=0.1)
            l1_loss = l1_loss * w_cfg["l1_weight"]

            loss = rmse_loss + tv_loss + l1_loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            log_meters["loss"] += loss.item()
            log_meters["rmse"] += rmse_loss.item()
            log_meters["tv"] += tv_loss.item()
            log_meters["l1"] += l1_loss.item()
            valid_batches += 1

        steps = max(valid_batches, 1)
        avg_train = {k: v / steps for k, v in log_meters.items()}

        # Validation
        model.eval()
        val_rmse_total = 0
        val_batches = 0
        with torch.no_grad():
            for x, y_lowres, mask_lowres in val_loader:
                x, y_lowres, mask_lowres = x.to(device), y_lowres.to(device), mask_lowres.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred_bias = model(x)
                    mask_hr = x[:, 168:169, :, :src_w]
                    forecast_hr = x[:, :168, :, :src_w]
                    pred_bias_hr = pred_bias[:, :, :, :src_w]
                    corrected_hr = forecast_hr - pred_bias_hr
                    mask_lowres = mask_lowres * grid_mask[None, :, :] * torch.from_numpy(glo12_ocean).float().to(device)[None, None, :, :]
                    if mask_lowres.sum().item() < 1.0:
                        continue
                    corrected_lowres = downsample_to_glo12(corrected_hr, grid, mask=mask_hr)

                v_loss = weighted_masked_rmse_loss(
                    corrected_lowres.float(), y_lowres.float(), mask_lowres.float(),
                    start_w=w_cfg["start_weight"], end_w=w_cfg["end_weight"],
                )
                if torch.isnan(v_loss):
                    continue
                val_rmse_total += v_loss.item()
                val_batches += 1

        avg_val_rmse = val_rmse_total / max(val_batches, 1)
        history_train_rmse.append(avg_train["rmse"])
        history_val_rmse.append(avg_val_rmse)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{trainer_cfg['num_epochs']} | "
              f"Val RMSE: {avg_val_rmse:.5f} | "
              f"Train RMSE: {avg_train['rmse']:.4f} "
              f"(L1: {avg_train['l1']:.4f}, TV: {avg_train['tv']:.4f}) | "
              f"LR: {lr:.2e}")

        scheduler.step()

        delta = trainer_cfg["early_stopping"].get("delta", 0.0)
        improved = avg_val_rmse < (best_val_loss - delta)
        if improved:
            best_val_loss = avg_val_rmse
            best_epoch = epoch
            early_stop_cnt = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> ✨ 性能提升！模型已保存: {save_path}")

            # 全图可视化（取一个 val 文件做推理）
            if trainer_cfg.get("vis_full_domain", True):
                _vis_full_map(model, val_loader, grid, grid_mask, glo12_ocean,
                              glo12, src_h, src_w, device, epoch, exp_cfg["save_dir"])
        else:
            early_stop_cnt += 1
            print(f"--> 💤 性能未提升 ({early_stop_cnt}/{trainer_cfg['early_stopping']['tolerance']})")

        if early_stop_cnt >= trainer_cfg["early_stopping"]["tolerance"]:
            print("🛑 触发早停机制，训练结束。")
            break

    if history_train_rmse and history_val_rmse:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(history_train_rmse)+1), history_train_rmse, label="Train RMSE")
        plt.plot(range(1, len(history_val_rmse)+1), history_val_rmse, label="Val RMSE")
        plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(exp_cfg["save_dir"], "training_curve.png"), dpi=150)
        plt.close()

    # Test
    print("\n🏆 开始在 Test Set 上评估 ...")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_rmse_total, test_mae_total, test_batches = 0, 0, 0
    with torch.no_grad():
        for x, y_lowres, mask_lowres in test_loader:
            x, y_lowres, mask_lowres = x.to(device), y_lowres.to(device), mask_lowres.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_bias = model(x)
                mask_hr = x[:, 168:169, :, :src_w]
                forecast_hr = x[:, :168, :, :src_w]
                pred_bias_hr = pred_bias[:, :, :, :src_w]
                corrected_hr = forecast_hr - pred_bias_hr
                mask_lowres = mask_lowres * grid_mask[None, :, :] * torch.from_numpy(glo12_ocean).float().to(device)[None, None, :, :]
                if mask_lowres.sum().item() < 1.0:
                    continue
                corrected_lowres = downsample_to_glo12(corrected_hr, grid, mask=mask_hr)

            cl = corrected_lowres.float(); yl = y_lowres.float(); ml = mask_lowres.float()
            t_loss = weighted_masked_rmse_loss(cl, yl, ml)
            mae = (torch.abs(cl - yl) * ml).sum() / (ml.sum() + 1e-6)
            if torch.isnan(t_loss):
                continue
            test_rmse_total += t_loss.item()
            test_mae_total += mae.item()
            test_batches += 1

    print(f"📊 Test RMSE: {test_rmse_total/max(test_batches,1):.5f}  "
          f"MAE: {test_mae_total/max(test_batches,1):.5f} °C")
    print(f"模型: {save_path}")


def _vis_full_map(model, val_loader, grid, grid_mask, glo12_ocean,
                  glo12, src_h, src_w, device, epoch, save_dir):
    """全图推理可视化：Forecast / Corrected / Target / Bias 对比。"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    model.eval()
    with torch.no_grad():
        for x, y_lowres, mask_lowres in val_loader:
            x = x.to(device); y_lowres = y_lowres.to(device); mask_lowres = mask_lowres.to(device)
            with torch.amp.autocast('cuda', enabled=True):
                pred_bias = model(x)
            forecast_hr = x[:, :168, :, :src_w]
            pred_bias_hr = pred_bias[:, :, :, :src_w]
            corrected_hr = forecast_hr - pred_bias_hr
            mask_hr = x[:, 168:169, :, :src_w]
            m = mask_lowres * grid_mask[None, :, :] * torch.from_numpy(glo12_ocean).float().to(device)[None, None, :, :]
            if m.sum() < 1:
                continue
            fc_lr = downsample_to_glo12(forecast_hr, grid, mask=mask_hr)[0].cpu().numpy()
            corr_lr = downsample_to_glo12(corrected_hr, grid, mask=mask_hr)[0].cpu().numpy()
            tgt = y_lowres[0].cpu().numpy()
            m = m[0].cpu().numpy()
            break

    for t in [0, 48, 96, 167]:
        if t >= fc_lr.shape[0]:
            continue
        fc_t = fc_lr[t].copy(); corr_t = corr_lr[t].copy(); tgt_t = tgt[t].copy()
        mt = m[t]
        fc_t[mt == 0] = np.nan; corr_t[mt == 0] = np.nan; tgt_t[mt == 0] = np.nan
        orig_b = fc_t - tgt_t; resid_b = corr_t - tgt_t
        orig_r = np.sqrt(np.nanmean(orig_b ** 2)); corr_r = np.sqrt(np.nanmean(resid_b ** 2))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        vmin = np.nanmin([np.nanmin(fc_t), np.nanmin(tgt_t)])
        vmax = np.nanmax([np.nanmax(fc_t), np.nanmax(tgt_t)])
        cmap = plt.cm.jet; cmap.set_bad('white')

        im1 = axes[0, 0].imshow(fc_t, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 0].set_title(f'Forecast T={t}h'); plt.colorbar(im1, ax=axes[0, 0])
        im2 = axes[0, 1].imshow(tgt_t, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 1].set_title('GLO12 Target'); plt.colorbar(im2, ax=axes[0, 1])
        im3 = axes[0, 2].imshow(corr_t, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 2].set_title(f'Corrected T={t}h'); plt.colorbar(im3, ax=axes[0, 2])

        bias_cmap = plt.cm.RdBu_r; bias_cmap.set_bad('white')
        bias_max = max(np.nanmax(np.abs(orig_b)), np.nanmax(np.abs(resid_b)), 0.1)
        im4 = axes[1, 0].imshow(orig_b, cmap=bias_cmap, vmin=-bias_max, vmax=bias_max, origin='lower')
        axes[1, 0].set_title(f'Original Bias (RMSE={orig_r:.3f})'); plt.colorbar(im4, ax=axes[1, 0])
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, f'RMSE\n{orig_r:.3f} → {corr_r:.3f}\n({(1-corr_r/orig_r)*100:.1f}%)',
                        ha='center', va='center', fontsize=16, fontweight='bold', transform=axes[1, 1].transAxes)
        im5 = axes[1, 2].imshow(resid_b, cmap=bias_cmap, vmin=-bias_max, vmax=bias_max, origin='lower')
        axes[1, 2].set_title(f'Residual Bias (RMSE={corr_r:.3f})'); plt.colorbar(im5, ax=axes[1, 2])

        plt.suptitle(f'Full-Map Val @ Epoch {epoch} T={t}h', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'vis_full_epoch{epoch:04d}_T{t:03d}.png'), dpi=150)
        plt.close()

    print(f"全图可视化已保存到 {save_dir}")
    model.train()


if __name__ == "__main__":
    run()
