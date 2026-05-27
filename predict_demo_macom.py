"""
MaCOM 订正结果可视化脚本。
读取 out_macom/*_corrected.nc + GLO12 数据，绘制对比图并保存 PNG。

用法:
    python predict_demo_macom.py --date 20260401 --step 48
    python predict_demo_macom.py --date 20260401 --step 48 --outdir ./demo_figures
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from netCDF4 import Dataset as NCDataset, num2date

import torch
import torch.nn.functional as F

from config import data_params
from data.glo12_reader import GLO12Reader
from dataset_macom import _extract_datetime_from_filename
from downsample import build_downsample_grid, downsample_to_glo12, downsample_mask_nearest


def find_corrected_file(date_str, corrected_dir="./out_macom"):
    """根据日期 YYYYMMDD 在 corrected_dir 中查找对应文件。"""
    # 新命名规范
    pattern_new = os.path.join(corrected_dir, f"AI_CResU-net_sstR_SH_{date_str}*_168.nc")
    matches = sorted(glob.glob(pattern_new))
    if matches:
        return matches[0]

    # 旧命名兼容
    for f in sorted(glob.glob(os.path.join(corrected_dir, "*_corrected.nc"))):
        if date_str in os.path.basename(f):
            return f

    raise FileNotFoundError(f"在 {corrected_dir} 中未找到日期 {date_str} 对应的订正文件")


def plot_with_target(forecast, corrected, bias, target, land_mask, date_str, t, save_dir):
    """有 GLO12 时的四宫格：Forecast / Corrected / Target / Bias。"""
    fc_t = forecast.copy()
    corr_t = corrected.copy()
    bias_t = bias.copy()
    tgt_t = target.copy()

    fc_t[land_mask] = np.nan
    corr_t[land_mask] = np.nan
    bias_t[land_mask] = np.nan
    tgt_t[land_mask] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    all_vals = np.concatenate([
        fc_t[~np.isnan(fc_t)], corr_t[~np.isnan(corr_t)],
        tgt_t[~np.isnan(tgt_t)],
    ])
    vmin = np.nanpercentile(np.asarray(all_vals), 2)
    vmax = np.nanpercentile(np.asarray(all_vals), 98)
    cmap = plt.cm.jet; cmap.set_bad('white')

    im1 = axes[0, 0].imshow(fc_t, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 0].set_title(f'Forecast T={t}h')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    im2 = axes[0, 1].imshow(corr_t, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 1].set_title(f'Corrected T={t}h')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

    im3 = axes[1, 0].imshow(tgt_t, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 0].set_title('GLO12 Target')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    bias_valid = bias_t[~np.isnan(bias_t)]
    abs_max = max(np.nanpercentile(np.abs(np.asarray(bias_valid)), 98), 0.1)
    bias_cmap = plt.cm.RdBu_r; bias_cmap.set_bad('white')
    bias_norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    im4 = axes[1, 1].imshow(bias_t, cmap=bias_cmap, norm=bias_norm, origin='lower')
    axes[1, 1].set_title(f'Predicted Bias T={t}h')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)

    plt.suptitle(f'SST Correction - {date_str} @ T={t}h', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{date_str}_T{t:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def plot_no_target(corrected, bias, land_mask, date_str, t, save_dir):
    """无 GLO12 时的 Fallback 图：1×2（Corrected / Bias），与 predict_demo_fvcom 一致。"""
    corr_t = corrected.copy()
    bias_t = bias.copy()
    corr_t[land_mask] = np.nan
    bias_t[land_mask] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    vmin = np.nanpercentile(np.asarray(corr_t), 2)
    vmax = np.nanpercentile(np.asarray(corr_t), 98)
    cmap = plt.cm.jet; cmap.set_bad('white')
    im1 = axes[0].imshow(corr_t, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(f'Corrected SST T={t}h')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    bias_valid = bias_t[~np.isnan(bias_t)]
    abs_max = max(np.nanpercentile(np.abs(np.asarray(bias_valid)), 98), 0.1)
    bias_cmap = plt.cm.RdBu_r; bias_cmap.set_bad('white')
    bias_norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    im2 = axes[1].imshow(bias_t, cmap=bias_cmap, norm=bias_norm, origin='lower')
    axes[1].set_title('Predicted Bias')
    plt.colorbar(im2, ax=axes[1], shrink=0.8, extend='both')

    plt.suptitle(f'SST Correction - {date_str} @ T={t}h', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{date_str}_T{t:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="MaCOM 订正结果可视化")
    parser.add_argument("--date", required=True, help="日期 YYYYMMDD，如 20260401")
    parser.add_argument("--step", type=int, default=0, help="预报时效（小时），默认 0")
    parser.add_argument("--corrected-dir", default="./out_macom",
                        help="订正结果目录（默认 ./out_macom）")
    parser.add_argument("--outdir", default="./out_macom/demo",
                        help="图片输出目录（默认 ./out_macom/demo）")
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    args = parser.parse_args()

    macom_cfg = data_params['macom']
    os.makedirs(args.outdir, exist_ok=True)

    # 1. 查找订正文件
    corrected_path = find_corrected_file(args.date, args.corrected_dir)
    print(f"订正文件: {corrected_path}")

    # 2. 读取订正结果 — 自动检测是 corrected_sst 还是 predicted_bias
    with NCDataset(corrected_path, "r") as ds:
        var_names = list(ds.variables.keys())
        src_lat = ds.variables['lat'][:]
        src_lon = ds.variables['lon'][:]

        if 'sst' in var_names:
            corr_full = ds.variables['sst'][:]
            var_mode = 'corrected'
        elif 'corrected_sst' in var_names:
            corr_full = ds.variables['corrected_sst'][:]
            var_mode = 'corrected'
        elif 'predicted_bias' in var_names:
            corr_full = ds.variables['predicted_bias'][:]
            var_mode = 'bias'
        else:
            raise KeyError(f"未识别的变量: {var_names}，期待 sst/corrected_sst 或 predicted_bias")
        print(f"  文件模式: {var_mode} ({'订正 SST' if var_mode == 'corrected' else '预测偏差'})")

    # 读取原始 MaCOM 文件来获取 forecast
    all_macom = sorted(glob.glob(macom_cfg['forecast_pattern']))
    matched_file = None
    for f in all_macom:
        if args.date in os.path.basename(f):
            matched_file = f
            break
    if matched_file is None:
        print("警告: 未找到匹配的原始 MaCOM 文件，无法获取 forecast。")
        return

    with NCDataset(matched_file, "r") as ds:
        var = ds.variables["t"]
        src_h, src_w = var.shape[2], var.shape[3]
        src_lat_orig = ds.variables["lat"][:]
        src_lon_orig = ds.variables["lon"][:]
        raw = var[:, 0, :, :]
        fc_raw = np.array(raw, dtype=np.float32)
        if np.ma.is_masked(raw):
            fc_raw[raw.mask] = np.nan

    # 截取到实际时效
    n_steps = min(corr_full.shape[0], fc_raw.shape[0])
    fc_full = fc_raw[:n_steps]
    corr_full = corr_full[:n_steps]

    # 根据文件模式计算另一个变量
    if var_mode == 'corrected':
        bias_full = fc_full - corr_full
    else:  # bias
        bias_full = corr_full.copy()
        corr_full = fc_full - bias_full

    src_h, src_w = len(src_lat), len(src_lon)
    t = min(args.step, n_steps - 1)
    print(f"网格: {src_h}x{src_w}, 时效: {n_steps}h, 绘制 T={t}h")

    # 3. 加载 GLO12 数据（若无 GLO12 文件则跳过对比）
    try:
        glo12 = GLO12Reader(macom_cfg['glo12_pattern'],
                             tolerance_hours=macom_cfg['time_tolerance_hours'])
        glo12_exists = True
    except FileNotFoundError:
        print("警告: GLO12 数据不可用，跳过低分辨率对比。")
        glo12_exists = False

    if not glo12_exists:
        land_mask = np.isnan(fc_full[0])
        plot_no_target(corr_full[t], bias_full[t], land_mask,
                       args.date, t, args.outdir)
        print(f"全部完成！图片保存在: {args.outdir}")
        return

    glo12_h, glo12_w = len(glo12.lat), len(glo12.lon)

    with NCDataset(matched_file, "r") as ds:
        tvar = ds.variables["time"]
        raw_time = tvar[:]
        time_units = tvar.units if "since" in tvar.units else "seconds since 1949-10-01 00:00:00"
        file_times = list(num2date(raw_time, units=time_units,
                                    calendar=getattr(tvar, "calendar", "standard")))

    glo12_indices, glo12_valid = glo12.nearest_time_indices(
        file_times, macom_cfg["time_tolerance_hours"])
    target = np.full((len(glo12_indices), glo12_h, glo12_w), np.nan, dtype=np.float32)
    if np.any(glo12_valid):
        target[glo12_valid] = glo12.get_sst(glo12_indices[glo12_valid])
    target = target[:n_steps]
    glo12_valid = glo12_valid[:n_steps]

    # 构建低分辨率 mask + downsample
    src_ocean = (~np.isnan(fc_full[0])).astype(np.float32)
    glo12_ocean = downsample_mask_nearest(src_ocean, src_lat, src_lon,
                                           glo12.lat, glo12.lon,
                                           y_offset=0, x_offset=0,
                                           patch_h=src_h, patch_w=src_w)

    grid = build_downsample_grid(src_lat, src_lon, src_h, src_w,
                                  glo12.lat, glo12.lon, glo12_h, glo12_w,
                                  y_offset=0, x_offset=0,
                                  patch_h=src_h, patch_w=src_w).to(args.device)
    grid_mask = (
        (grid[0, :, :, 0] >= -1.0) & (grid[0, :, :, 0] <= 1.0) &
        (grid[0, :, :, 1] >= -1.0) & (grid[0, :, :, 1] <= 1.0)
    ).cpu().numpy().astype(np.float32)

    mask_hr = torch.from_numpy(src_ocean).float().to(args.device)
    fc_tensor = torch.from_numpy(fc_full).float().unsqueeze(0).to(args.device)
    corr_tensor = torch.from_numpy(corr_full).float().unsqueeze(0).to(args.device)
    grid = grid.to(args.device)

    fc_lr = downsample_to_glo12(fc_tensor, grid, mask=mask_hr)[0].cpu().numpy()
    corr_lr = downsample_to_glo12(corr_tensor, grid, mask=mask_hr)[0].cpu().numpy()

    glo12_land = np.any(~np.isnan(target), axis=0).astype(np.float32)
    time_mask = glo12_valid.astype(np.float32)
    mask_lowres = time_mask[:, None, None] * glo12_land[None, :, :] * glo12_ocean[None, :, :] * grid_mask[None, :, :]

    # 高分辨率 GLO12 投影
    glo12_lat_step = (glo12.lat[-1] - glo12.lat[0]) / (glo12_h - 1)
    glo12_lon_step = (glo12.lon[-1] - glo12.lon[0]) / (glo12_w - 1)
    glo12_y = (src_lat - glo12.lat[0]) / glo12_lat_step
    glo12_x = (src_lon - glo12.lon[0]) / glo12_lon_step
    norm_y = 2.0 * glo12_y / (glo12_h - 1) - 1.0
    norm_x = 2.0 * glo12_x / (glo12_w - 1) - 1.0
    grid_gx, grid_gy = np.meshgrid(norm_x, norm_y)
    grid_g = np.stack([grid_gx, grid_gy], axis=-1)
    grid_g = torch.from_numpy(grid_g).float().unsqueeze(0).to(args.device)
    target_tensor = torch.from_numpy(target).float().unsqueeze(0).to(args.device)
    glo12_hr = F.grid_sample(target_tensor, grid_g, mode='nearest', align_corners=True)[0].cpu().numpy()

    # 4. 将 GLO12 投影到高分辨率网格，用于与 Corrected / Bias 同尺寸对比
    if target is not None:
        glo12_lat_step = (glo12.lat[-1] - glo12.lat[0]) / (glo12_h - 1)
        glo12_lon_step = (glo12.lon[-1] - glo12.lon[0]) / (glo12_w - 1)
        glo12_y = (src_lat - glo12.lat[0]) / glo12_lat_step
        glo12_x = (src_lon - glo12.lon[0]) / glo12_lon_step
        norm_y = 2.0 * glo12_y / (glo12_h - 1) - 1.0
        norm_x = 2.0 * glo12_x / (glo12_w - 1) - 1.0
        grid_gx, grid_gy = np.meshgrid(norm_x, norm_y)
        grid_g = np.stack([grid_gx, grid_gy], axis=-1)
        grid_g = torch.from_numpy(grid_g).float().unsqueeze(0).to(args.device)
        target_tensor = torch.from_numpy(target).float().unsqueeze(0).to(args.device)
        target_hr = F.grid_sample(target_tensor, grid_g, mode='nearest', align_corners=True)[0].cpu().numpy()
    else:
        target_hr = None

    # 5. 绘图
    land_mask = np.isnan(fc_full[0])

    if target_hr is not None:
        plot_with_target(fc_full[t], corr_full[t], bias_full[t], target_hr[t],
                         land_mask, args.date, t, args.outdir)
    else:
        plot_no_target(corr_full[t], bias_full[t], land_mask,
                       args.date, t, args.outdir)

    print(f"全部完成！图片保存在: {args.outdir}")


if __name__ == "__main__":
    main()
