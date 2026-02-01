import numpy as np
import torch
from netCDF4 import Dataset as NCDataset
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from config import experiment_params, model_params
from models.baseline.CResU_Net import CRUNet


def _build_model(device):
    model_cfg = model_params['CResU_Net']
    model = CRUNet(
        in_channels=model_cfg['core']['in_channels'],
        out_channels=model_cfg['core']['out_channels'],
        selected_dim=0,
        device=device
    ).to(device)
    return model


def _load_land_mask(forecast_path):
    with NCDataset(forecast_path, 'r') as ds:
        raw_mask = ds.variables['land_mask'][:]
    return np.nan_to_num(raw_mask, nan=0.0)[np.newaxis, :, :]


def correct_run_from_nc(model_path, forecast_path, run_idx=0, device=None, ignore_steps=None):
    """
    读取某一天的 120 小时预报，输出订正后的 120 小时结果。
    返回: corrected_sst, pred_bias
    """
    if device is None:
        device = torch.device(experiment_params['device'] if torch.cuda.is_available() else 'cpu')
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    model = _build_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    land_mask = _load_land_mask(forecast_path)
    with NCDataset(forecast_path, 'r') as ds:
        sst_raw = ds.variables['sst'][run_idx]
    sst_input = np.nan_to_num(sst_raw, nan=0.0)

    x = np.concatenate((sst_input, land_mask), axis=0)
    x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_bias = model(x_tensor)

    if ignore_steps is None:
        ignore_steps = model_params['CResU_Net']['trainer']['loss_weights'].get('ignore_steps', 0)
    if ignore_steps and ignore_steps > 0:
        ignore_steps = min(ignore_steps, pred_bias.shape[1])
        pred_bias[:, :ignore_steps] = 0.0

    corrected = x_tensor[0, :120] - pred_bias[0]
    return corrected.cpu().numpy(), pred_bias[0].cpu().numpy()


def visualize_run_step(
    forecast_path,
    reanalysis_path,
    run_idx,
    t,
    corrected_sst,
    pred_bias,
    mask_land=True,
):
    """
    可视化单个时间步的原始场、订正场、原始偏差与订正残差。
    """
    with NCDataset(forecast_path, 'r') as ds:
        forecast = ds.variables['sst'][run_idx][t]
        forecast = np.nan_to_num(forecast, nan=0.0)
        land_mask = ds.variables['land_mask'][:]
        ocean_mask = np.nan_to_num(land_mask, nan=0.0)
        fc_time = float(np.round(ds.variables['valid_time'][run_idx][t], 2))

    with NCDataset(reanalysis_path, 'r') as ds:
        ra_times = ds.variables['time'][:]
        ra_map = {round(float(x), 2): i for i, x in enumerate(ra_times)}
        ra_idx = ra_map.get(fc_time)
        if ra_idx is None:
            raise ValueError(f"未在再分析数据中找到时间 {fc_time} 的匹配项")
        target = ds.variables['sst'][ra_idx]
        target = np.nan_to_num(target, nan=0.0)

    corrected = corrected_sst[t].copy()
    bias = pred_bias[t].copy()

    if mask_land:
        forecast[ocean_mask == 0] = np.nan
        corrected[ocean_mask == 0] = np.nan
        bias[ocean_mask == 0] = np.nan
        target[ocean_mask == 0] = np.nan

    orig_bias = forecast - target
    res_bias = corrected - target

    orig_rmse = np.sqrt(np.nanmean(orig_bias ** 2))
    res_rmse = np.sqrt(np.nanmean(res_bias ** 2))

    bias_abs_max = np.nanmax(np.abs(np.concatenate([orig_bias.ravel(), res_bias.ravel()])))
    if bias_abs_max == 0:
        bias_abs_max = 0.1
    bias_norm = TwoSlopeNorm(vmin=-bias_abs_max, vcenter=0.0, vmax=bias_abs_max)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Forecast")
    plt.imshow(forecast, origin="lower", cmap="jet")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("Corrected")
    plt.imshow(corrected, origin="lower", cmap="jet")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title(f"Forecast Bias\nRMSE: {orig_rmse:.4f}°C")
    plt.imshow(orig_bias, origin="lower", cmap="seismic", norm=bias_norm)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title(f"Residual Bias\nRMSE: {res_rmse:.4f}°C")
    plt.imshow(res_bias, origin="lower", cmap="seismic", norm=bias_norm)
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def compute_run_rmse(
    forecast_path,
    reanalysis_path,
    run_idx,
    corrected_sst,
):
    """
    统计某一天 120 小时的 RMSE，并打印原始 vs 订正对比。
    返回: (rmse_forecast, rmse_corrected)
    """
    with NCDataset(forecast_path, 'r') as fc_ds:
        fc_times = fc_ds.variables['valid_time'][run_idx]
        fc_all = fc_ds.variables['sst'][run_idx]
        fc_all = np.nan_to_num(fc_all, nan=0.0)
        land_mask = fc_ds.variables['land_mask'][:]
        ocean_mask = np.nan_to_num(land_mask, nan=0.0)

    with NCDataset(reanalysis_path, 'r') as ra_ds:
        ra_times = ra_ds.variables['time'][:]
        ra_map = {round(float(x), 2): i for i, x in enumerate(ra_times)}

        time_pairs = [(t, ra_map.get(round(float(fc_time), 2))) for t, fc_time in enumerate(fc_times)]
        time_pairs = [(t, idx) for t, idx in time_pairs if idx is not None]

        if len(time_pairs) == 0:
            raise ValueError("没有找到可用的时间步用于 RMSE 统计")

        t_indices = [t for t, _ in time_pairs]
        ra_indices = [idx for _, idx in time_pairs]

        target_all = ra_ds.variables['sst'][ra_indices]
        target_all = np.nan_to_num(target_all, nan=0.0)

    fc_sel = fc_all[t_indices]
    corr_sel = corrected_sst[t_indices]

    fc_sel = fc_sel.copy()
    corr_sel = corr_sel.copy()
    target_all = target_all.copy()

    fc_sel[:, ocean_mask == 0] = np.nan
    corr_sel[:, ocean_mask == 0] = np.nan
    target_all[:, ocean_mask == 0] = np.nan

    rmse_forecast = np.sqrt(np.nanmean((fc_sel - target_all) ** 2))
    rmse_corrected = np.sqrt(np.nanmean((corr_sel - target_all) ** 2))
    print(f"RMSE (Forecast): {rmse_forecast:.5f}°C")
    print(f"RMSE (Corrected): {rmse_corrected:.5f}°C")
    return rmse_forecast, rmse_corrected


def compute_yearly_error(
    model_path,
    forecast_path,
    reanalysis_path,
    device=None,
    batch_size=4,
    mask_initial_steps=True,
):
    """
    统计全年 RMSE/MAE（Forecast vs Corrected）。
    返回: ((rmse_forecast, mae_forecast), (rmse_corrected, mae_corrected))
    """
    if device is None:
        device = torch.device(experiment_params['device'] if torch.cuda.is_available() else 'cpu')
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    model = _build_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    ignore_steps = model_params['CResU_Net']['trainer']['loss_weights'].get('ignore_steps', 0)

    sum_abs_fc = 0.0
    sum_abs_corr = 0.0
    sum_sq_fc = 0.0
    sum_sq_corr = 0.0
    count = 0

    with NCDataset(forecast_path, 'r') as fc_ds, NCDataset(reanalysis_path, 'r') as ra_ds:
        ra_times = ra_ds.variables['time'][:]
        ra_map = {round(float(x), 2): i for i, x in enumerate(ra_times)}

        run_count = fc_ds.dimensions['run'].size
        land_mask = fc_ds.variables['land_mask'][:]
        ocean_mask = np.nan_to_num(land_mask, nan=0.0)

        for start in range(0, run_count, batch_size):
            end = min(start + batch_size, run_count)
            batch_indices = list(range(start, end))

            fc_batch = fc_ds.variables['sst'][batch_indices]
            fc_batch = np.nan_to_num(fc_batch, nan=0.0)

            mask_channel = np.broadcast_to(ocean_mask, (len(batch_indices), 1) + ocean_mask.shape)
            x = np.concatenate((fc_batch, mask_channel), axis=1)

            x_tensor = torch.from_numpy(x).float().to(device)
            with torch.no_grad():
                pred_bias = model(x_tensor)
            if mask_initial_steps and ignore_steps and ignore_steps > 0:
                pred_bias[:, :ignore_steps] = 0.0

            corrected = (x_tensor[:, :120] - pred_bias).cpu().numpy()

            for b, run_idx in enumerate(batch_indices):
                fc_times = fc_ds.variables['valid_time'][run_idx]
                time_pairs = [(t, ra_map.get(round(float(fc_time), 2))) for t, fc_time in enumerate(fc_times)]
                time_pairs = [(t, idx) for t, idx in time_pairs if idx is not None]
                if len(time_pairs) == 0:
                    continue

                t_indices = [t for t, _ in time_pairs]
                ra_indices = [idx for _, idx in time_pairs]

                target = ra_ds.variables['sst'][ra_indices]
                target = np.nan_to_num(target, nan=0.0)

                fc_sel = fc_batch[b, t_indices].copy()
                corr_sel = corrected[b, t_indices].copy()
                target = target.copy()

                fc_sel[:, ocean_mask == 0] = np.nan
                corr_sel[:, ocean_mask == 0] = np.nan
                target[:, ocean_mask == 0] = np.nan

                diff_fc = fc_sel - target
                diff_corr = corr_sel - target

                sum_abs_fc += np.nansum(np.abs(diff_fc))
                sum_abs_corr += np.nansum(np.abs(diff_corr))
                sum_sq_fc += np.nansum(diff_fc ** 2)
                sum_sq_corr += np.nansum(diff_corr ** 2)
                count += np.sum(~np.isnan(target))

    if count == 0:
        raise ValueError("没有找到可用的时间步用于误差统计")

    mae_fc = sum_abs_fc / count
    mae_corr = sum_abs_corr / count
    rmse_fc = np.sqrt(sum_sq_fc / count)
    rmse_corr = np.sqrt(sum_sq_corr / count)

    print(f"RMSE (Forecast): {rmse_fc:.5f} °C")
    print(f"MAE  (Forecast): {mae_fc:.5f} °C")
    print(f"RMSE (Corrected): {rmse_corr:.5f} °C")
    print(f"MAE  (Corrected): {mae_corr:.5f} °C")

    return (rmse_fc, mae_fc), (rmse_corr, mae_corr)


def compute_yearly_mae(model_path, forecast_path, reanalysis_path, device=None, batch_size=4):
    """
    兼容旧接口：仅返回 MAE。
    """
    (rmse_fc, mae_fc), (rmse_corr, mae_corr) = compute_yearly_error(
        model_path=model_path,
        forecast_path=forecast_path,
        reanalysis_path=reanalysis_path,
        device=device,
        batch_size=batch_size,
        mask_initial_steps=True,
    )
    return mae_fc, mae_corr
