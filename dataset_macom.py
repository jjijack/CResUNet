"""
MaCOM 全图数据集。
返回完整 MaCOM 网格（1168×1008，pad 到 16 的倍数），
不做 patch 切分——模型直接学习全局空间上下文，消除拼接痕迹。
"""

import glob
import os
import re
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as NCDataset, num2date

from data.glo12_reader import GLO12Reader


def _extract_datetime_from_filename(file_path):
    """从 MaCOM 文件名中提取日期时间。"""
    basename = os.path.basename(file_path)
    candidates = re.findall(r"\d{8,14}", basename)
    if not candidates:
        return None
    token = max(candidates, key=len)
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H", "%Y%m%d"):
        try:
            return datetime.strptime(token[:len(datetime.now().strftime(fmt))], fmt)
        except ValueError:
            continue
    try:
        return datetime.strptime(token[:8], "%Y%m%d")
    except ValueError:
        return None


class MaCOMPatchDataset(Dataset):
    """MaCOM 全图数据集 — 每个文件返回完整网格（兼容旧名称）。

    返回:
        x: (171, H, W_pad) — 168 SST + 1 mask + 2 position encoding
        y: (168, glo12_h, glo12_w) — GLO12 目标
        mask: (168, glo12_h, glo12_w) — 有效区域 mask
    """

    def __init__(
        self,
        forecast_pattern,
        glo12_pattern,
        glo12_reader=None,
        target_steps=168,
        time_tolerance_hours=0.6,
        start_date=None,
        end_date=None,
    ):
        super().__init__()
        self.target_steps = target_steps
        self.time_tolerance_hours = time_tolerance_hours

        all_files = sorted(glob.glob(forecast_pattern))
        self.files = []
        for f in all_files:
            dt = _extract_datetime_from_filename(f)
            if dt is None:
                continue
            if start_date and dt < start_date:
                continue
            if end_date and dt >= end_date:
                continue
            self.files.append(f)

        if not self.files:
            raise FileNotFoundError(f"No MaCOM files matched: {forecast_pattern}")

        if glo12_reader is not None:
            self.glo12 = glo12_reader
        else:
            self.glo12 = GLO12Reader(glo12_pattern, tolerance_hours=time_tolerance_hours)
        self.glo12_h = len(self.glo12.lat)
        self.glo12_w = len(self.glo12.lon)

        with NCDataset(self.files[0], "r") as ds:
            var = ds.variables["t"]
            self.src_h, self.src_w = var.shape[2], var.shape[3]
            self.src_lat = ds.variables["lat"][:]
            self.src_lon = ds.variables["lon"][:]

            # 预计算海洋掩膜（用于 GLO12 downsampling mask）
            data_slice = var[:, 0, :, :]
            sst0 = np.array(data_slice, dtype=np.float32)
            if np.ma.is_masked(data_slice):
                sst0[data_slice.mask] = np.nan
            self.ocean_mask = (~np.isnan(sst0)).any(axis=0).astype(np.float32)  # (H, W)

        self.pad_w = (16 - self.src_w % 16) % 16
        self.padded_w = self.src_w + self.pad_w

        # Pad ocean mask
        if self.pad_w > 0:
            self.ocean_mask = np.pad(self.ocean_mask, ((0, 0), (0, self.pad_w)), mode="edge")

        lat_2d = np.broadcast_to(self.src_lat[:, None], (self.src_h, self.src_w)).astype(np.float32)
        lon_2d = np.broadcast_to(self.src_lon[None, :], (self.src_h, self.src_w)).astype(np.float32)
        if self.pad_w > 0:
            lon_2d = np.pad(lon_2d, ((0, 0), (0, self.pad_w)), mode="edge")
            lat_2d = np.pad(lat_2d, ((0, 0), (0, self.pad_w)), mode="edge")
        self.lat_map = 2.0 * (lat_2d - lat_2d.min()) / (lat_2d.max() - lat_2d.min() + 1e-8) - 1.0
        self.lon_map = 2.0 * (lon_2d - lon_2d.min()) / (lon_2d.max() - lon_2d.min() + 1e-8) - 1.0

        print(f"MaCOMPatchDataset (全图模式): {len(self.files)} files, "
              f"grid={self.src_h}x{self.src_w}->{self.src_h}x{self.padded_w}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]

        with NCDataset(f, "r") as ds:
            tvar = ds.variables["time"]
            raw_time = tvar[:]
            time_units = tvar.units
            if "since" not in time_units:
                long_name = getattr(tvar, "long_name", "")
                if "since" in long_name:
                    time_units = long_name
                else:
                    time_units = "seconds since 1949-10-01 00:00:00"
            dt_objs = num2date(raw_time, units=time_units, calendar=getattr(tvar, "calendar", "standard"))
            file_times = list(dt_objs)
            glo12_indices, glo12_valid = self.glo12.nearest_time_indices(file_times, self.time_tolerance_hours)

            data = ds.variables["t"][:, 0, :, :]
            sst = np.array(data, dtype=np.float32)
            if np.ma.is_masked(data):
                sst[data.mask] = np.nan

        n_file_steps = sst.shape[0]

        if self.pad_w > 0:
            sst = np.pad(sst, ((0, 0), (0, 0), (0, self.pad_w)), mode="edge")

        land_mask = (~np.isnan(sst)).any(axis=0).astype(np.float32)

        nan_frac = np.isnan(sst[:, :, :self.src_w]).mean(axis=(1, 2))
        macom_bad_steps = nan_frac > 0.9

        glo12_sst = np.full((len(glo12_indices), self.glo12_h, self.glo12_w), np.nan, dtype=np.float32)
        if np.any(glo12_valid):
            glo12_sst[glo12_valid] = self.glo12.get_sst(glo12_indices[glo12_valid])

        n_steps = min(n_file_steps, self.target_steps)
        sst = sst[:n_steps]
        glo12_sst = glo12_sst[:n_steps]
        glo12_valid = glo12_valid[:n_steps] & ~macom_bad_steps[:n_steps]

        if n_steps < self.target_steps:
            pad_len = self.target_steps - n_steps
            sst = np.pad(sst, ((0, pad_len), (0, 0), (0, 0)), mode="constant", constant_values=0.0)
            glo12_sst = np.pad(glo12_sst, ((0, pad_len), (0, 0), (0, 0)), mode="constant", constant_values=np.nan)
            glo12_valid = np.pad(glo12_valid, (0, pad_len), mode="constant", constant_values=False)

        sst_input = np.nan_to_num(sst, nan=0.0)

        mask_ch = land_mask.reshape(1, self.src_h, self.padded_w)
        lat_ch = self.lat_map.reshape(1, self.src_h, self.padded_w)
        lon_ch = self.lon_map.reshape(1, self.src_h, self.padded_w)
        x = np.concatenate([sst_input, mask_ch, lat_ch, lon_ch], axis=0)

        glo12_land = np.any(~np.isnan(glo12_sst), axis=0).astype(np.float32)
        time_mask = glo12_valid.astype(np.float32)
        valid_mask = time_mask[:, None, None] * glo12_land[None, :, :]

        glo12_sst = np.nan_to_num(glo12_sst, nan=0.0)

        if valid_mask.shape[0] < self.target_steps:
            pad_len = self.target_steps - valid_mask.shape[0]
            valid_mask = np.pad(valid_mask, ((0, pad_len), (0, 0), (0, 0)), mode="constant", constant_values=0.0)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(glo12_sst).float(),
            torch.from_numpy(valid_mask).float(),
        )
