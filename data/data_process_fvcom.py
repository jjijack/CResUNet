import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import matplotlib.tri as mtri
import numpy as np
from netCDF4 import Dataset as NCDataset, num2date
from tqdm import tqdm

from data_process_utils import (
    create_adaptive_triangulation,
    detect_sst_var,
    init_forecast_nc,
    init_reanalysis_nc,
    list_files_with_date_filter,
    parse_datetime_input,
    write_forecast_run,
    write_reanalysis_block,
)


worker_triang = None
worker_mesh_lon = None
worker_mesh_lat = None


def _init_worker(x, y, triangles, mask, mesh_lon, mesh_lat):
    global worker_triang, worker_mesh_lon, worker_mesh_lat
    worker_triang = mtri.Triangulation(x, y, triangles=triangles)
    if mask is not None:
        worker_triang.set_mask(mask)
    worker_mesh_lon = mesh_lon
    worker_mesh_lat = mesh_lat


def _process_file_return_data(nc_file):
    global worker_triang, worker_mesh_lon, worker_mesh_lat
    if worker_triang is None:
        return None

    try:
        with NCDataset(nc_file, "r") as ds:
            sst_var = detect_sst_var(ds)
            data_raw = ds.variables[sst_var][:]
            if data_raw.ndim == 3:
                data_raw = data_raw[:, 0, :]

            if "time" in ds.variables:
                tvar = ds.variables["time"]
                raw_time = tvar[:]
                if np.ma.is_masked(raw_time):
                    raw_time = raw_time.data
                try:
                    time_objs = num2date(raw_time, units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
                except Exception:
                    time_objs = [datetime(2000, 1, 1)] * data_raw.shape[0]
            else:
                time_objs = [datetime(2000, 1, 1)] * data_raw.shape[0]

            out = np.full((data_raw.shape[0], worker_mesh_lat.shape[0], worker_mesh_lat.shape[1]), np.nan, dtype=np.float32)
            for step in range(data_raw.shape[0]):
                interpolator = mtri.LinearTriInterpolator(worker_triang, data_raw[step])
                out[step] = interpolator(worker_mesh_lon, worker_mesh_lat).filled(np.nan)
            return out, time_objs
    except Exception as exc:
        print(f"Error processing {os.path.basename(nc_file)}: {exc}")
        return None


def process_source(
    source_name,
    pattern,
    mode,
    output_path,
    mesh_lat,
    mesh_lon,
    fixed_steps,
    start_date,
    end_date,
    adaptive_k,
    adaptive_tol,
):
    files = list_files_with_date_filter(pattern, start_date=start_date, end_date=end_date)
    if not files:
        print(f"[Skip] {source_name}: no matched files")
        return

    print(f"\n=== 开始处理: {source_name} (模式: {mode}, 文件数: {len(files)}) ===")
    print(f"正在构建自适应拓扑 (Reference: {os.path.basename(files[0])})...")

    triang = create_adaptive_triangulation(files[0], adaptive_k=adaptive_k, adaptive_tol=adaptive_tol)
    print(f"拓扑构建完毕: 剔除 {int(np.sum(triang.mask)) if triang.mask is not None else 0} 个非法三角形 (去扇形)")

    mask_interp = mtri.LinearTriInterpolator(triang, np.ones(triang.x.shape))
    valid_mask = mask_interp(mesh_lon, mesh_lat).filled(0.0).astype(np.int8)

    if mode == "stack":
        nc = init_forecast_nc(output_path, mesh_lat, mesh_lon, valid_mask, fixed_steps=fixed_steps)
    else:
        nc = init_reanalysis_nc(output_path, mesh_lat, mesh_lon, valid_mask)

    try:
        curr_idx = 0
        print("启动并行处理...")
        init_args = (triang.x, triang.y, triang.triangles, triang.mask, mesh_lon, mesh_lat)
        with ProcessPoolExecutor(initializer=_init_worker, initargs=init_args) as executor:
            for result in tqdm(executor.map(_process_file_return_data, files), total=len(files)):
                if result is None:
                    continue
                block_data, block_times = result

                if mode == "stack":
                    write_forecast_run(nc, curr_idx, block_data, block_times, fixed_steps=fixed_steps)
                    curr_idx += 1
                else:
                    curr_idx += write_reanalysis_block(nc, curr_idx, block_data, block_times)

        print(f"✅ 处理完成! 结果已保存至: {output_path}")
    finally:
        nc.close()


def main():
    parser = argparse.ArgumentParser(description="Process raw SST files into structured forecast/reanalysis NetCDF")
    parser.add_argument("--forecast-pattern", required=True, help="Glob pattern for raw forecast files")
    parser.add_argument("--reanalysis-pattern", required=True, help="Glob pattern for raw reanalysis files")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    parser.add_argument("--start-date", default=None, help="Inclusive start datetime")
    parser.add_argument("--end-date", default=None, help="Inclusive end datetime")

    parser.add_argument("--save-forecast", action="store_true", default=True, help="Generate forecast_structured.nc")
    parser.add_argument("--save-reanalysis", action="store_true", help="Generate reanalysis_structured.nc")

    parser.add_argument("--fixed-steps", type=int, default=120, help="Fixed steps for forecast stack mode")
    parser.add_argument("--target-h", type=int, default=608)
    parser.add_argument("--target-w", type=int, default=704)
    parser.add_argument("--lon-min", type=float, default=117.50)
    parser.add_argument("--lon-max", type=float, default=124.55)
    parser.add_argument("--lat-min", type=float, default=28.30)
    parser.add_argument("--lat-max", type=float, default=34.40)
    parser.add_argument("--adaptive-k", type=int, default=8)
    parser.add_argument("--adaptive-tol", type=float, default=2.0)
    args = parser.parse_args()

    start_date = parse_datetime_input(args.start_date)
    end_date = parse_datetime_input(args.end_date)
    if start_date and end_date and start_date > end_date:
        raise ValueError("--start-date cannot be later than --end-date")

    os.makedirs(args.output_dir, exist_ok=True)

    grid_lon = np.linspace(args.lon_min, args.lon_max, args.target_w)
    grid_lat = np.linspace(args.lat_min, args.lat_max, args.target_h)
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    if args.save_forecast:
        process_source(
            source_name="forecast",
            pattern=args.forecast_pattern,
            mode="stack",
            output_path=os.path.join(args.output_dir, "forecast_structured.nc"),
            mesh_lat=mesh_lat,
            mesh_lon=mesh_lon,
            fixed_steps=args.fixed_steps,
            start_date=start_date,
            end_date=end_date,
            adaptive_k=args.adaptive_k,
            adaptive_tol=args.adaptive_tol,
        )

    if args.save_reanalysis:
        process_source(
            source_name="reanalysis",
            pattern=args.reanalysis_pattern,
            mode="concat",
            output_path=os.path.join(args.output_dir, "reanalysis_structured.nc"),
            mesh_lat=mesh_lat,
            mesh_lon=mesh_lon,
            fixed_steps=args.fixed_steps,
            start_date=start_date,
            end_date=end_date,
            adaptive_k=args.adaptive_k,
            adaptive_tol=args.adaptive_tol,
        )


if __name__ == "__main__":
    main()
