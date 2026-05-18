"""
MaCOM 数据处理脚本
将 MaCOM 的 swt（SST）数据从规则网格插值到与 FVCOM 完全相同的网格，
输出格式与 data_process.py 生成的 forecast_structured.nc 完全一致。
"""

import argparse
import os
import re
from datetime import datetime

import numpy as np
from netCDF4 import Dataset as NCDataset, date2num
from tqdm import tqdm

# 与 data_process_utils.py 保持一致
TIME_UNITS = "hours since 1970-01-01 00:00:00"


# ===================== 工具函数 =====================

def parse_datetime_input(value):
    if value is None or value == "":
        return None
    candidates = [
        "%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
        "%Y%m%d", "%Y%m%d%H", "%Y%m%d%H%M", "%Y%m%d%H%M%S",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid datetime: {value}") from exc


def _extract_datetime_from_filename(file_path):
    """从文件名中提取日期时间，如 MaCOM_swt_SH_001h_20260404_168.nc -> 20260404"""
    basename = os.path.basename(file_path)
    candidates = re.findall(r"\d{8,14}", basename)
    if not candidates:
        return None
    token = max(candidates, key=len)
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H", "%Y%m%d"):
        try:
            if len(token) == len(datetime.now().strftime(fmt)):
                return datetime.strptime(token, fmt)
        except ValueError:
            continue
    token = token[:8]
    try:
        return datetime.strptime(token, "%Y%m%d")
    except ValueError:
        return None


def list_files_with_date_filter(pattern, start_date=None, end_date=None):
    import glob
    all_files = sorted(glob.glob(pattern))
    if start_date is None and end_date is None:
        return all_files
    selected = []
    for file_path in all_files:
        dt = _extract_datetime_from_filename(file_path)
        if dt is None:
            continue
        if start_date is not None and dt < start_date:
            continue
        if end_date is not None and dt > end_date:
            continue
        selected.append(file_path)
    return selected


def read_macom_swt(nc_file):
    """读取 MaCOM swt 文件，返回 (sst_data, time_objs)
    sst_data: (n_time, lat, lon) float32, 单位 Celsius
    time_objs: list of datetime
    """
    from netCDF4 import num2date

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


def init_forecast_nc(filename, tgt_lat, tgt_lon, fixed_steps):
    """创建输出 NetCDF，格式与 data_process_utils.init_forecast_nc 完全一致"""
    nc = NCDataset(filename, "w", format="NETCDF4")
    nc.createDimension("run", None)
    nc.createDimension("step", fixed_steps)
    nc.createDimension("lat", len(tgt_lat))
    nc.createDimension("lon", len(tgt_lon))

    nc.createVariable("lat", "f4", ("lat",))[:] = tgt_lat
    nc.createVariable("lon", "f4", ("lon",))[:] = tgt_lon
    # MaCOM 无 land_mask，所有点标记为有效（因为后续 predict 会用 mask 通道，设为 0 即全海洋）
    nc.createVariable("land_mask", "i1", ("lat", "lon"), zlib=True)[:] = 0

    sst = nc.createVariable("sst", "f4", ("run", "step", "lat", "lon"), zlib=True, fill_value=np.nan)
    sst.description = "SST Forecast (Stacked by Run) - MaCOM"

    v_start = nc.createVariable("start_time", "f8", ("run",))
    v_start.units = TIME_UNITS
    v_start.calendar = "standard"

    v_valid = nc.createVariable("valid_time", "f8", ("run", "step"))
    v_valid.units = TIME_UNITS
    v_valid.calendar = "standard"

    return nc


def write_forecast_run(nc, run_idx, block_data, block_times, fixed_steps):
    """写入一个 run 的数据，与 data_process_utils.write_forecast_run 完全一致"""
    final_data = np.full((fixed_steps, block_data.shape[1], block_data.shape[2]), np.nan, dtype=np.float32)
    final_times = [block_times[0]] * fixed_steps

    use_len = min(block_data.shape[0], fixed_steps)
    final_data[:use_len] = block_data[:use_len]
    final_times[:use_len] = block_times[:use_len]

    nc.variables["sst"][run_idx, :, :, :] = final_data

    units = nc.variables["start_time"].units
    calendar = getattr(nc.variables["start_time"], "calendar", "standard")
    nc.variables["start_time"][run_idx] = date2num(final_times[0], units=units, calendar=calendar)
    nc.variables["valid_time"][run_idx, :] = date2num(final_times, units=units, calendar=calendar)


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(
        description="Process MaCOM swt files into structured forecast NetCDF (same grid as FVCOM)"
    )
    parser.add_argument("--forecast-pattern", required=True,
                        help="Glob pattern for MaCOM swt files, e.g. /path/to/*_001h_*.nc")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    parser.add_argument("--start-date", default=None, help="Inclusive start datetime")
    parser.add_argument("--end-date", default=None, help="Inclusive end datetime")

    parser.add_argument("--fixed-steps", type=int, default=120,
                        help="Fixed steps for forecast stack (default: 120)")
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

    # 2. 列出并筛选文件
    files = list_files_with_date_filter(args.forecast_pattern, start_date=start_date, end_date=end_date)
    if not files:
        print("[Skip] No matched MaCOM files")
        return

    print(f"\n=== MaCOM 数据处理 ===")
    print(f"目标网格: {args.target_h}x{args.target_w}, "
          f"lat=[{args.lat_min},{args.lat_max}], lon=[{args.lon_min},{args.lon_max}]")
    print(f"文件数: {len(files)}, fixed_steps={args.fixed_steps}")
    print(f"日期范围: {start_date or 'ALL'} ~ {end_date or 'ALL'}")

    # 3. 读取第一个文件获取源网格坐标（所有文件共享同一网格）
    _, _, src_lon, src_lat = read_macom_swt(files[0])
    print(f"源网格: {len(src_lat)}x{len(src_lon)}, "
          f"lat=[{src_lat[0]:.4f},{src_lat[-1]:.4f}], lon=[{src_lon[0]:.4f},{src_lon[-1]:.4f}]")

    # 4. 预计算插值索引（只算一次，所有文件复用）
    print("预计算插值索引...")
    i0, j0, w_lat, w_lon = build_interp_indices(src_lat, src_lon, tgt_lat, tgt_lon)

    # 5. 打开输出文件
    output_path = os.path.join(args.output_dir, "forecast_structured.nc")
    nc_out = init_forecast_nc(output_path, tgt_lat, tgt_lon, args.fixed_steps)

    try:
        run_idx = 0
        for nc_file in tqdm(files, desc="Processing MaCOM files"):
            try:
                sst_data, time_objs, _, _ = read_macom_swt(nc_file)
            except Exception as e:
                print(f"Error reading {os.path.basename(nc_file)}: {e}")
                continue

            # 向量化双线性插值（所有时步一次完成）
            interp_data = batch_bilinear_interp(sst_data, i0, j0, w_lat, w_lon)

            # 写入为一个 run（每个 MaCOM 文件 = 一个 run）
            write_forecast_run(nc_out, run_idx, interp_data, list(time_objs), args.fixed_steps)
            run_idx += 1

        print(f"✅ 处理完成! {run_idx} 个 runs 已写入: {output_path}")
    finally:
        nc_out.close()


if __name__ == "__main__":
    main()
