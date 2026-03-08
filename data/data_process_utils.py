import os
import re
from datetime import datetime

import matplotlib.tri as mtri
import numpy as np
from netCDF4 import Dataset as NCDataset, date2num, num2date
from scipy.spatial import cKDTree


TIME_UNITS = "hours since 1970-01-01 00:00:00"


def parse_datetime_input(value):
    if value is None or value == "":
        return None

    candidates = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%d",
        "%Y%m%d%H",
        "%Y%m%d%H%M",
        "%Y%m%d%H%M%S",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid datetime: {value}. Supported examples: 2026-01-01, 2026-01-01 12:00, 2026010112"
        ) from exc


def _extract_datetime_from_filename(file_path):
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
    all_files = [f for f in all_files if "stationsout" not in os.path.basename(f)]

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


def detect_sst_var(ds):
    for var_name in ("sst", "temp", "temperature", "sea_surface_temperature"):
        if var_name in ds.variables:
            return var_name
    raise ValueError("No SST-like variable found. Tried: sst/temp/temperature/sea_surface_temperature")


def create_adaptive_triangulation(reference_file, adaptive_k=8, adaptive_tol=2.0):
    with NCDataset(reference_file, "r") as ds:
        x = ds.variables["lon"][:]
        y = ds.variables["lat"][:]

    points = np.column_stack([x, y])
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=adaptive_k + 1)
    local_scale = np.mean(dists[:, 1:], axis=1)

    triang = mtri.Triangulation(x, y)
    triangles = triang.triangles

    scale_tri = local_scale[triangles]
    ref_scale = np.mean(scale_tri, axis=1)

    xt, yt = x[triangles], y[triangles]
    edge0 = np.hypot(xt[:, 0] - xt[:, 1], yt[:, 0] - yt[:, 1])
    edge1 = np.hypot(xt[:, 1] - xt[:, 2], yt[:, 1] - yt[:, 2])
    edge2 = np.hypot(xt[:, 2] - xt[:, 0], yt[:, 2] - yt[:, 0])
    max_edge = np.max(np.vstack([edge0, edge1, edge2]), axis=0)

    mask = max_edge > (ref_scale * adaptive_tol)
    triang.set_mask(mask)
    return triang


def interpolate_one_file(nc_file, triang, mesh_lon, mesh_lat):
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

    out = np.full((data_raw.shape[0], mesh_lat.shape[0], mesh_lat.shape[1]), np.nan, dtype=np.float32)
    for step in range(data_raw.shape[0]):
        interpolator = mtri.LinearTriInterpolator(triang, data_raw[step])
        out[step] = interpolator(mesh_lon, mesh_lat).filled(np.nan)
    return out, time_objs


def init_forecast_nc(filename, mesh_lat, mesh_lon, mask, fixed_steps):
    nc = NCDataset(filename, "w", format="NETCDF4")
    nc.createDimension("run", None)
    nc.createDimension("step", fixed_steps)
    nc.createDimension("lat", mesh_lat.shape[0])
    nc.createDimension("lon", mesh_lon.shape[1])

    nc.createVariable("lat", "f4", ("lat",))[:] = mesh_lat[:, 0]
    nc.createVariable("lon", "f4", ("lon",))[:] = mesh_lon[0, :]
    nc.createVariable("land_mask", "i1", ("lat", "lon"), zlib=True)[:] = mask

    sst = nc.createVariable("sst", "f4", ("run", "step", "lat", "lon"), zlib=True, fill_value=np.nan)
    sst.description = "SST Forecast (Stacked by Run)"

    v_start = nc.createVariable("start_time", "f8", ("run",))
    v_start.units = TIME_UNITS
    v_start.calendar = "standard"

    v_valid = nc.createVariable("valid_time", "f8", ("run", "step"))
    v_valid.units = TIME_UNITS
    v_valid.calendar = "standard"

    return nc


def init_reanalysis_nc(filename, mesh_lat, mesh_lon, mask):
    nc = NCDataset(filename, "w", format="NETCDF4")
    nc.createDimension("time", None)
    nc.createDimension("lat", mesh_lat.shape[0])
    nc.createDimension("lon", mesh_lon.shape[1])

    nc.createVariable("lat", "f4", ("lat",))[:] = mesh_lat[:, 0]
    nc.createVariable("lon", "f4", ("lon",))[:] = mesh_lon[0, :]
    nc.createVariable("land_mask", "i1", ("lat", "lon"), zlib=True)[:] = mask

    nc.createVariable("sst", "f4", ("time", "lat", "lon"), zlib=True, fill_value=np.nan)
    t = nc.createVariable("time", "f8", ("time",))
    t.units = TIME_UNITS
    t.calendar = "standard"
    return nc


def write_forecast_run(nc, run_idx, block_data, block_times, fixed_steps):
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


def write_reanalysis_block(nc, time_offset, block_data, block_times):
    n = block_data.shape[0]
    nc.variables["sst"][time_offset: time_offset + n] = block_data

    units = nc.variables["time"].units
    calendar = getattr(nc.variables["time"], "calendar", "standard")
    t_nums = date2num(block_times, units=units, calendar=calendar)
    nc.variables["time"][time_offset: time_offset + n] = t_nums
    return n
