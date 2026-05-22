"""
MaCOM 数据处理脚本
将 MaCOM 的 swt（SST）数据从规则网格插值到与 FVCOM 完全相同的网格，
输出格式与 data_process.py 生成的 forecast_structured.nc 完全一致。
"""

import argparse
import os

import numpy as np
from tqdm import tqdm
from data_process_utils import (
    init_forecast_nc,
    init_reanalysis_nc,
    list_files_with_date_filter,
    parse_datetime_input,
    write_forecast_run,
    write_reanalysis_block,
)
from netCDF4 import Dataset as NCDataset, num2date


# ===================== 读取 MaCOM 数据 =====================

def read_macom_swt(nc_file):
    """读取 MaCOM swt 或 reanalysis 文件，返回 (sst_data, time_objs, src_lon, src_lat)
    sst_data: (n_time, lat, lon) float32, 单位 Celsius
    time_objs: list of datetime
    """
    with NCDataset(nc_file, "r") as ds:
        # MaCOM swt 文件变量名为 "t"，维度为 (time, z, lat, lon)
        var = ds.variables["t"]
        data = var[:]  # masked array
        # squeeze z 维度（z=1）
        if data.ndim == 4:
            data = data[:, 0, :, :]
        # 转为 float32，填充值变 NaN
        sst_data = np.array(data, dtype=np.float32)
        if np.ma.is_masked(data):
            sst_data[data.mask] = np.nan

        # 读取时间
        tvar = ds.variables["time"]
        raw_time = tvar[:]
        if np.ma.is_masked(raw_time):
            raw_time = raw_time.data
        # MaCOM 的 time:units 仅为 "seconds"，参考日期在 long_name 中
        # 如 long_name="seconds since 1949-10-01 00:00:00"
        time_units = tvar.units
        if "since" not in time_units:
            long_name = getattr(tvar, "long_name", "")
            if "since" in long_name:
                time_units = long_name
            else:
                time_units = "seconds since 1949-10-01 00:00:00"
        time_objs = num2date(
            raw_time,
            units=time_units,
            calendar=getattr(tvar, "calendar", "standard"),
        )

        # 读取源网格坐标
        src_lon = ds.variables["lon"][:]
        src_lat = ds.variables["lat"][:]

    return sst_data, time_objs, src_lon, src_lat


# ===================== 插值与输出 =====================

def build_interp_indices(src_lat, src_lon, tgt_lat, tgt_lon):
    """预计算目标网格在源网格中的索引和权重，用于向量化双线性插值。
    返回 (i0, j0, w_lat, w_lon)，其中:
      i0, j0: 左上角索引 (tgt_h, tgt_w)
      w_lat, w_lon: 插值权重 (tgt_h, tgt_w)
    """
    # 经度索引
    lon_step = src_lon[1] - src_lon[0]
    lon_frac = (tgt_lon - src_lon[0]) / lon_step
    j0 = np.floor(lon_frac).astype(int)
    j0 = np.clip(j0, 0, len(src_lon) - 2)
    w_lon = (lon_frac - j0).astype(np.float32)
    w_lon = np.clip(w_lon, 0.0, 1.0)

    # 纬度索引
    lat_step = src_lat[1] - src_lat[0]
    lat_frac = (tgt_lat - src_lat[0]) / lat_step
    i0 = np.floor(lat_frac).astype(int)
    i0 = np.clip(i0, 0, len(src_lat) - 2)
    w_lat = (lat_frac - i0).astype(np.float32)
    w_lat = np.clip(w_lat, 0.0, 1.0)

    # 扩展为 2D 网格 (tgt_h, tgt_w)
    i0_2d = np.broadcast_to(i0[:, None], (len(tgt_lat), len(tgt_lon)))
    j0_2d = np.broadcast_to(j0[None, :], (len(tgt_lat), len(tgt_lon)))
    w_lat_2d = np.broadcast_to(w_lat[:, None], (len(tgt_lat), len(tgt_lon)))
    w_lon_2d = np.broadcast_to(w_lon[None, :], (len(tgt_lat), len(tgt_lon)))

    return i0_2d, j0_2d, w_lat_2d, w_lon_2d


def batch_bilinear_interp(src_data, i0, j0, w_lat, w_lon):
    """向量化双线性插值，一次处理所有时步。
    src_data: (n_time, src_h, src_w)
    i0, j0, w_lat, w_lon: (tgt_h, tgt_w) 预计算的索引和权重
    返回: (n_time, tgt_h, tgt_w)
    """
    n_time = src_data.shape[0]

    # 取四个角的值: (n_time, tgt_h, tgt_w)
    v00 = src_data[:, i0, j0]
    v01 = src_data[:, i0, j0 + 1]
    v10 = src_data[:, i0 + 1, j0]
    v11 = src_data[:, i0 + 1, j0 + 1]

    # 双线性插值
    w_lat = w_lat[None, :, :]  # (1, tgt_h, tgt_w)
    w_lon = w_lon[None, :, :]
    result = (1 - w_lat) * (1 - w_lon) * v00 + \
             (1 - w_lat) * w_lon * v01 + \
             w_lat * (1 - w_lon) * v10 + \
             w_lat * w_lon * v11

    return result.astype(np.float32)


def process_macom_source(
    source_name,
    pattern,
    mode,
    output_path,
    tgt_lat,
    tgt_lon,
    fixed_steps,
    start_date,
    end_date,
):
    """处理 MaCOM 数据源（forecast 或 reanalysis），插值到目标网格并输出 structured nc。
    mode: "stack" (forecast, run+step) 或 "concat" (reanalysis, 拼接时间序列)
    """
    files = list_files_with_date_filter(pattern, start_date=start_date, end_date=end_date)
    if not files:
        print(f"[Skip] {source_name}: no matched MaCOM files")
        return

    print(f"源网格读取中...")
    _, _, src_lon, src_lat = read_macom_swt(files[0])
    print(f"源网格: {len(src_lat)}x{len(src_lon)}, "
          f"lat=[{src_lat[0]:.4f},{src_lat[-1]:.4f}], lon=[{src_lon[0]:.4f},{src_lon[-1]:.4f}]")

    print("预计算插值索引...")
    i0, j0, w_lat, w_lon = build_interp_indices(src_lat, src_lon, tgt_lat, tgt_lon)

    # 从第一个文件推导 land_mask（NaN=陆地，有效=海洋）
    sst_first, _, _, _ = read_macom_swt(files[0])
    interp_first = batch_bilinear_interp(sst_first[:1], i0, j0, w_lat, w_lon)
    land_mask = (~np.isnan(interp_first[0])).astype(np.int8)
    print(f"Land mask: {land_mask.sum()} ocean / {land_mask.size} total ({land_mask.sum()/land_mask.size:.1%})")

    if mode == "stack":
        nc_out = init_forecast_nc(output_path, tgt_lat, tgt_lon, land_mask, fixed_steps)
    else:
        nc_out = init_reanalysis_nc(output_path, tgt_lat, tgt_lon, land_mask)

    try:
        if mode == "stack":
            run_idx = 0
            for nc_file in tqdm(files, desc=f"Processing {source_name}"):
                try:
                    sst_data, time_objs, _, _ = read_macom_swt(nc_file)
                except Exception as e:
                    print(f"Error reading {os.path.basename(nc_file)}: {e}")
                    continue
                interp_data = batch_bilinear_interp(sst_data, i0, j0, w_lat, w_lon)
                write_forecast_run(nc_out, run_idx, interp_data, list(time_objs), fixed_steps)
                run_idx += 1
            print(f"✅ {source_name} 处理完成! {run_idx} 个 runs 已写入: {output_path}")
        else:
            curr_idx = 0
            for nc_file in tqdm(files, desc=f"Processing {source_name}"):
                try:
                    sst_data, time_objs, _, _ = read_macom_swt(nc_file)
                except Exception as e:
                    print(f"Error reading {os.path.basename(nc_file)}: {e}")
                    continue
                interp_data = batch_bilinear_interp(sst_data, i0, j0, w_lat, w_lon)
                curr_idx += write_reanalysis_block(nc_out, curr_idx, interp_data, list(time_objs))
            print(f"✅ {source_name} 处理完成! {curr_idx} 个时步已写入: {output_path}")
    finally:
        nc_out.close()


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(
        description="Process MaCOM swt files into structured forecast NetCDF (same grid as FVCOM)"
    )
    parser.add_argument("--forecast-pattern", required=True,
                        help="Glob pattern for MaCOM swt files, e.g. /path/to/*_001h_*.nc")
    parser.add_argument("--reanalysis-pattern", default=None,
                        help="Glob pattern for MaCOM reanalysis files (optional)")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    parser.add_argument("--start-date", default=None, help="Inclusive start datetime")
    parser.add_argument("--end-date", default=None, help="Inclusive end datetime")

    parser.add_argument("--save-forecast", action="store_true", default=True,
                        help="Generate forecast_structured.nc")
    parser.add_argument("--save-reanalysis", action="store_true",
                        help="Generate reanalysis_structured.nc")

    parser.add_argument("--fixed-steps", type=int, default=168,
                        help="Fixed steps for forecast stack (default: 168 for MaCOM)")
    # 目标网格参数——与 data_process.py 默认值保持一致
    parser.add_argument("--target-h", type=int, default=608)
    parser.add_argument("--target-w", type=int, default=704)
    parser.add_argument("--lon-min", type=float, default=117.50)
    parser.add_argument("--lon-max", type=float, default=124.55)
    parser.add_argument("--lat-min", type=float, default=28.30)
    parser.add_argument("--lat-max", type=float, default=34.40)
    args = parser.parse_args()

    start_date = parse_datetime_input(args.start_date)
    end_date = parse_datetime_input(args.end_date)
    if start_date and end_date and start_date > end_date:
        raise ValueError("--start-date cannot be later than --end-date")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 构建目标网格（与 forecast_structured.nc 一致）
    tgt_lon = np.linspace(args.lon_min, args.lon_max, args.target_w)
    tgt_lat = np.linspace(args.lat_min, args.lat_max, args.target_h)

    # --- 处理 forecast ---
    if args.save_forecast:
        print("\n=== MaCOM forecast 数据处理 ===")
        process_macom_source(
            source_name="forecast",
            pattern=args.forecast_pattern,
            mode="stack",
            output_path=os.path.join(args.output_dir, "forecast_structured.nc"),
            tgt_lat=tgt_lat,
            tgt_lon=tgt_lon,
            fixed_steps=args.fixed_steps,
            start_date=start_date,
            end_date=end_date,
        )

    # --- 处理 reanalysis ---
    if args.save_reanalysis:
        if not args.reanalysis_pattern:
            print("[Skip] --save-reanalysis requires --reanalysis-pattern")
        else:
            print("\n=== MaCOM reanalysis 数据处理 ===")
            process_macom_source(
                source_name="reanalysis",
                pattern=args.reanalysis_pattern,
                mode="concat",
                output_path=os.path.join(args.output_dir, "reanalysis_structured.nc"),
                tgt_lat=tgt_lat,
                tgt_lon=tgt_lon,
                fixed_steps=args.fixed_steps,
                start_date=start_date,
                end_date=end_date,
            )


if __name__ == "__main__":
    main()
