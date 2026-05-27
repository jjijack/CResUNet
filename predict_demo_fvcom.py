import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date


def main():
    parser = argparse.ArgumentParser(description="SST Correction Visualization Tool")
    parser.add_argument("--run-idx", type=int, default=0, help="Index of the forecast run (0 = 1st day)")
    parser.add_argument("--step", type=int, default=24, help="Forecast lead time in hours (0-167 for MaCOM, 0-119 for FVCOM)")
    args = parser.parse_args()

    # 1. 路径配置（自动检测项目根目录）
    proj_root = os.path.dirname(os.path.abspath(__file__))
    forecast_path = os.path.join(proj_root, "data/forecast_structured.nc")
    corrected_path = os.path.join(proj_root, "out/forecast_corrected_structured.nc")
    reanalysis_path = os.path.join(proj_root, "data/reanalysis_structured.nc")
    out_dir = os.path.join(proj_root, "out/demo")
    os.makedirs(out_dir, exist_ok=True)

    # 2. 读取原始预报
    with Dataset(forecast_path, 'r') as ds:
        st_var = ds.variables['start_time']
        start_dt = num2date(st_var[args.run_idx], units=st_var.units, calendar=getattr(st_var, 'calendar', 'standard'))

        vt_var = ds.variables['valid_time']
        target_val = vt_var[args.run_idx, args.step]
        valid_dt = num2date(target_val, units=vt_var.units, calendar=getattr(vt_var, 'calendar', 'standard'))

        raw_fc = ds.variables['sst'][args.run_idx, args.step]
        land_mask = ds.variables['land_mask'][:]

    # 3. 读取订正结果（直接从已生成的 nc 文件）
    with Dataset(corrected_path, 'r') as ds:
        corr_sst = ds.variables['sst'][args.run_idx, args.step]
        if 'pred_bias' in ds.variables:
            p_bias = ds.variables['pred_bias'][args.run_idx, args.step]
        else:
            p_bias = corr_sst - raw_fc

    print(f">>> Run: {args.run_idx}, Step: {args.step}h")
    print(f"    Start: {start_dt} | Valid: {valid_dt}")

    # 4. 应用陆地掩码
    ocean_idx = (land_mask == 1)
    raw_fc_plot = np.where(ocean_idx, raw_fc, np.nan)
    corr_sst_plot = np.where(ocean_idx, corr_sst, np.nan)
    p_bias_plot = np.where(ocean_idx, p_bias, np.nan)

    # 5. 读取监督数据（如有）
    target_plot = None
    if os.path.exists(reanalysis_path):
        with Dataset(reanalysis_path, 'r') as ds_ra:
            ra_times = ds_ra.variables['time'][:]
            ra_idx = np.where(np.abs(ra_times - target_val) < 1e-5)[0]
            if len(ra_idx) > 0:
                target_plot = ds_ra.variables['sst'][ra_idx[0]]
                target_plot = np.where(ocean_idx, target_plot, np.nan)

    # 6. 绘图
    fig_title = f"SST Correction Analysis\nStart: {start_dt.strftime('%Y-%m-%d %H:%M')} | Lead: +{args.step}h ({valid_dt.strftime('%Y-%m-%d %H:%M')})"

    bias_max = 5.0

    if target_plot is not None:
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))

        orig_bias = raw_fc_plot - target_plot
        res_bias = corr_sst_plot - target_plot
        error_reduction = np.abs(orig_bias) - np.abs(res_bias)

        o_rmse = np.sqrt(np.nanmean(orig_bias**2))
        c_rmse = np.sqrt(np.nanmean(res_bias**2))
        improvement_pct = (o_rmse - c_rmse) / o_rmse * 100

        im1 = axes[0].imshow(corr_sst_plot, origin='lower', cmap='jet')
        axes[0].set_title(f"Corrected SST\n(RMSE: {c_rmse:.4f}degC)")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(p_bias_plot, origin='lower', cmap='seismic', vmin=-bias_max, vmax=bias_max)
        axes[1].set_title("Predicted Bias\n(Forecast - Corrected)")
        plt.colorbar(im2, ax=axes[1], extend='both')

        im3 = axes[2].imshow(error_reduction, origin='lower', cmap='seismic', vmin=-bias_max, vmax=bias_max)
        axes[2].set_title(f"Error Reduction Map\n(Improvement: {improvement_pct:.2f}%)")
        plt.colorbar(im3, ax=axes[2], extend='both')

        print(f"✅ 统计完成: 原始 RMSE {o_rmse:.4f} -> 订正 RMSE {c_rmse:.4f} (整体提升了 {improvement_pct:.2f}%)")

    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        im1 = axes[0].imshow(corr_sst_plot, origin='lower', cmap='jet')
        axes[0].set_title("Corrected SST")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(p_bias_plot, origin='lower', cmap='seismic', vmin=-bias_max, vmax=bias_max)
        axes[1].set_title("Predicted Bias\n(Forecast - Corrected)")
        plt.colorbar(im2, ax=axes[1], extend='both')

    plt.suptitle(fig_title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_name = f"demo_r{args.run_idx}_s{args.step}.png"
    save_path = os.path.join(out_dir, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 成功: 可视化图片已保存至 {save_path}")

if __name__ == "__main__":
    main()