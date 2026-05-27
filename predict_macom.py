"""
MaCOM 全图推理脚本。
直接读原始 MaCOM .nc 文件，做全图推理，输出订正后的高分辨率 SST。
"""

import os
import glob
import argparse
from datetime import timedelta
import numpy as np
import torch
from tqdm import tqdm
from netCDF4 import Dataset as NCDataset, num2date, date2num

from config import experiment_params, data_params, model_params
from models.baseline.CResU_Net import CRUNet
from dataset_macom import _extract_datetime_from_filename
from data.glo12_reader import GLO12Reader


def build_model(device):
    model_cfg = model_params['CResU_Net']
    model = CRUNet(
        in_channels=model_cfg['core']['in_channels'],
        out_channels=model_cfg['core']['out_channels'],
        selected_dim=0,
        device=device,
        base_channels=model_cfg['core'].get('base_channels', 64),
        dropout=model_cfg['core'].get('dropout', 0.0),
    ).to(device)
    return model


def predict_one_file(model, nc_path, device, target_steps=168, pad_multiple=16, pos_scale=1.0):
    """对单个 MaCOM 文件做全图推理，返回 (fc_full, corr_full, bias_full, src_lat, src_lon, file_datetime)。"""
    with NCDataset(nc_path, "r") as ds:
        var = ds.variables["t"]
        src_h, src_w = var.shape[2], var.shape[3]
        src_lat = ds.variables["lat"][:]
        src_lon = ds.variables["lon"][:]
        tvar = ds.variables["time"]
        raw_time = tvar[:]
        time_units = tvar.units if "since" in tvar.units else "seconds since 1949-10-01 00:00:00"
        calendar = getattr(tvar, "calendar", "standard")
        data = ds.variables["t"][:, 0, :, :]
        sst = np.array(data, dtype=np.float32)
        if np.ma.is_masked(data):
            sst[data.mask] = np.nan

    # 提取文件起始时间的 datetime 对象
    dt_objs = num2date(raw_time, units=time_units, calendar=calendar)
    start_datetime = dt_objs[0]

    real_steps = min(sst.shape[0], target_steps)
    sst_crop = sst[:real_steps]

    # 在 nan_to_num 之前计算海洋掩膜（原始数据中陆地=NaN）
    ocean_mask_2d = (~np.isnan(sst_crop)).any(axis=0).astype(np.float32)  # (H, W)

    # Pad 宽度到 16 的倍数
    pad_w = (pad_multiple - src_w % pad_multiple) % pad_multiple
    padded_w = src_w + pad_w
    if pad_w > 0:
        sst_crop = np.pad(sst_crop, ((0, 0), (0, 0), (0, pad_w)), mode="edge")
        ocean_mask_2d = np.pad(ocean_mask_2d, ((0, 0), (0, pad_w)), mode="edge")

    # 位置编码
    lat_2d = np.broadcast_to(src_lat[:, None], (src_h, src_w)).astype(np.float32)
    lon_2d = np.broadcast_to(src_lon[None, :], (src_h, src_w)).astype(np.float32)
    if pad_w > 0:
        lat_2d = np.pad(lat_2d, ((0, 0), (0, pad_w)), mode="edge")
        lon_2d = np.pad(lon_2d, ((0, 0), (0, pad_w)), mode="edge")
    lat_map = pos_scale * (2.0 * (lat_2d - lat_2d.min()) / (lat_2d.max() - lat_2d.min() + 1e-8) - 1.0)
    lon_map = pos_scale * (2.0 * (lon_2d - lon_2d.min()) / (lon_2d.max() - lon_2d.min() + 1e-8) - 1.0)

    land_mask = ocean_mask_2d  # 用于模型输入

    if real_steps < target_steps:
        pad_len = target_steps - real_steps
        sst_crop = np.pad(sst_crop, ((0, pad_len), (0, 0), (0, 0)), mode="constant", constant_values=0.0)

    sst_input = np.nan_to_num(sst_crop, nan=0.0)

    # 组装输入 (171, H, W_pad)
    mask_ch = land_mask.reshape(1, src_h, padded_w)
    lat_ch = lat_map.reshape(1, src_h, padded_w)
    lon_ch = lon_map.reshape(1, src_h, padded_w)
    x_np = np.concatenate([sst_input, mask_ch, lat_ch, lon_ch], axis=0)
    x_tensor = torch.from_numpy(x_np).float().unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            pred_bias = model(x_tensor)

    corrected = x_tensor[:, :target_steps] - pred_bias
    fc_full = x_tensor[0, :target_steps, :, :src_w].cpu().float().numpy()
    corr_full = corrected[0, :, :, :src_w].cpu().float().numpy()
    bias_full = pred_bias[0, :, :, :src_w].cpu().float().numpy()

    # 海洋区域置 NaN（使用原始数据计算的掩膜，避免 np.nan_to_num 干扰）
    ocean_3d = ocean_mask_2d[None, :, :src_w]  # (1, H, W)
    fc_full = np.where(ocean_3d, fc_full, np.nan)
    corr_full = np.where(ocean_3d, corr_full, np.nan)
    bias_full = np.where(ocean_3d, bias_full, np.nan)

    return fc_full, corr_full, bias_full, src_lat, src_lon, start_datetime


def save_output_nc(output_path, output_sst, src_lat, src_lon, start_datetime, target_steps, save_bias, basename):
    """保存推理结果为 NetCDF 文件，遵循 CF 元数据规范。

    Args:
        output_sst: (T, H, W) 待保存的场（已 mask 陆地）
        save_bias: True 时保存 predicted_bias，否则保存 corrected_sst
    """
    t_steps = output_sst.shape[0]

    # 变量定义
    if save_bias:
        var_name = 'predicted_bias'
        var_long_name = 'Model-predicted SST bias'
        var_units = '°C'
    else:
        var_name = 'sst'
        var_long_name = 'Corrected Sea Surface Temperature'
        var_units = '°C'

    with NCDataset(output_path, 'w', format='NETCDF4') as dst:
        dst.createDimension('time', t_steps)
        dst.createDimension('lat', len(src_lat))
        dst.createDimension('lon', len(src_lon))

        # 时间：Unix 纪元（使用 date2num 兼容 cftime 对象）
        time_vals = np.array([
            date2num(start_datetime + timedelta(hours=h),
                     units='hours since 1970-01-01 00:00:00',
                     calendar='standard')
            for h in range(t_steps)
        ], dtype=np.float64)
        t_var = dst.createVariable('time', 'f8', ('time',))
        t_var.units = 'hours since 1970-01-01 00:00:00'
        t_var.calendar = 'standard'
        t_var.long_name = 'Time'
        t_var[:] = time_vals

        lat_var = dst.createVariable('lat', 'f4', ('lat',))
        lat_var.units = 'degrees_north'
        lat_var.long_name = 'Latitude'
        lat_var.axis = 'Y'
        lat_var[:] = src_lat

        lon_var = dst.createVariable('lon', 'f4', ('lon',))
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'Longitude'
        lon_var.axis = 'X'
        lon_var[:] = src_lon

        sst_var = dst.createVariable(var_name, 'f4', ('time', 'lat', 'lon'),
                                     zlib=True, fill_value=np.nan,
                                     least_significant_digit=3)
        sst_var.long_name = var_long_name
        sst_var.units = var_units
        sst_var[:] = output_sst

        # 全局属性
        dst.title = f"AI_CResU-net_sstR_SH Corrected SST"
        dst.institution = "AI"
        dst.source = "CResU-net"
        dst.history = f"Generated by predict_macom.py at {np.datetime64('now')}"
        dst.Conventions = "CF-1.8"
    print(f"  已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MaCOM 全图推理：输出订正后的高分辨率 SST。")
    parser.add_argument("--model", required=True, help="模型 .pth 文件路径")
    parser.add_argument("--forecast-pattern", default=None,
                        help="MaCOM 文件 glob 模式（默认从 config 读取）")
    parser.add_argument("--output-dir", default="./out_macom",
                        help="输出目录（默认 ./out_macom）")
    parser.add_argument("--device", default=None, help="cuda/cpu")
    parser.add_argument("--start-date", default=None, help="起始日期 (含)，如 2026-04-01")
    parser.add_argument("--end-date", default=None, help="结束日期 (不含)，如 2026-05-01")
    parser.add_argument("--save-bias", action="store_true",
                        help="输出 predicted_bias 而非 corrected_sst（默认输出订正 SST）")
    args = parser.parse_args()

    macom_cfg = data_params['macom']
    forecast_pattern = args.forecast_pattern or macom_cfg['forecast_pattern']
    target_steps = macom_cfg['target_steps']

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # 收集文件并按日期筛选
    all_files = sorted(glob.glob(forecast_pattern))
    if not all_files:
        raise FileNotFoundError(f"没有匹配到 MaCOM 文件: {forecast_pattern}")

    if args.start_date or args.end_date:
        from predict_utils import parse_datetime_input
        start_dt = parse_datetime_input(args.start_date)
        end_dt = parse_datetime_input(args.end_date)
        filtered = []
        for f in all_files:
            file_dt = _extract_datetime_from_filename(f)
            if file_dt is None:
                continue
            if start_dt and file_dt < start_dt:
                continue
            if end_dt and file_dt >= end_dt:
                continue
            filtered.append(f)
        all_files = filtered
        print(f"日期筛选后: {len(all_files)} 个文件")

    print(f"待推理文件数: {len(all_files)}")

    # 加载模型
    model = build_model(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f"模型加载完成: {args.model}")

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    for file_path in tqdm(all_files, desc="推理进度"):
        basename = os.path.basename(file_path)

        try:
            fc, corr, bias, src_lat, src_lon, start_dt = \
                predict_one_file(model, file_path, device, target_steps=target_steps)
        except Exception as e:
            print(f"  推理失败 {basename}: {e}")
            continue

        # 文件名: AI_CResU-net_sstR_SH_YYYYMMDDHH_168.nc（直接放在输出目录）
        file_time_str = start_dt.strftime("%Y%m%d%H")
        out_filename = f"AI_CResU-net_sstR_SH_{file_time_str}_{target_steps}.nc"
        output_path = os.path.join(args.output_dir, out_filename)

        # 选择输出变量（默认覆盖已有文件）
        output_sst = bias if args.save_bias else corr
        save_output_nc(output_path, output_sst, src_lat, src_lon, start_dt,
                       target_steps, args.save_bias, basename)

    print(f"全部完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
