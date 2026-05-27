"""
GLO12 再分析数据读取器
支持单文件/多文件，提供最近邻时间匹配（含容差）。
用于 macom 新管线的低分辨率监督。
"""

import glob
import os
from datetime import datetime

import numpy as np
from netCDF4 import Dataset as NCDataset, num2date


class GLO12Reader:
    """GLO12 再分析数据读取器（支持多文件拼接）。

    用法:
        reader = GLO12Reader("/path/to/glo12/*.nc")
        idx = reader.nearest_time_index(datetime(2026, 4, 5, 0, 32))  # 容差内返回 idx
        sst = reader.get_sst([idx])  # (1, lat, lon)
    """

    def __init__(self, pattern, tolerance_hours=0.6):
        """
        Args:
            pattern: glob 模式，如 "/path/to/GLO12/*.nc"
            tolerance_hours: 时间匹配容差（小时），默认 0.6
        """
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No GLO12 files found: {pattern}")
        self.tolerance_hours = tolerance_hours
        self._tolerance_seconds = tolerance_hours * 3600.0

        # 读取所有文件的时间戳，建立索引
        self._times_dt = []       # list of datetime
        self._times_hours = []    # float64, hours since 1970-01-01
        self._file_indices = []   # (file_idx, local_idx) 映射
        self._files_open = False

        for fidx, fpath in enumerate(self.files):
            with NCDataset(fpath, "r") as ds:
                tvar = ds.variables["time"]
                raw = tvar[:]
                units = tvar.units
                calendar = getattr(tvar, "calendar", "standard")
                dt_objs = num2date(raw, units=units, calendar=calendar)

                for local_idx, dt in enumerate(dt_objs):
                    self._times_dt.append(dt)
                    # 转为 hours since 1970-01-01 统一单位
                    epoch = datetime(1970, 1, 1)
                    delta = dt - epoch
                    hours = delta.total_seconds() / 3600.0
                    self._times_hours.append(hours)
                    self._file_indices.append((fidx, local_idx))

        self._times_hours = np.array(self._times_hours, dtype=np.float64)

        # 读取 lat/lon（从第一个文件）
        with NCDataset(self.files[0], "r") as ds:
            self.lat = ds.variables["latitude"][:]
            self.lon = ds.variables["longitude"][:]

        # 缓存：用于加速多次读取
        self._sst_cache = {}  # fpath -> numpy array

    @property
    def shape(self):
        """返回 (n_time, n_lat, n_lon)"""
        return (len(self._times_hours), len(self.lat), len(self.lon))

    def nearest_time_index(self, dt, tolerance_hours=None):
        """返回与 dt 最近的时间索引，超出容差返回 None。

        Args:
            dt: datetime 对象
            tolerance_hours: 覆盖默认容差（小时）

        Returns:
            int 或 None
        """
        tol = (tolerance_hours or self.tolerance_hours) * 3600.0
        epoch = datetime(1970, 1, 1)
        delta = dt - epoch
        target_hours = delta.total_seconds() / 3600.0

        diffs = np.abs(self._times_hours - target_hours)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= tol:
            return int(min_idx)
        return None

    def nearest_time_indices(self, dts, tolerance_hours=None):
        """批量最近邻时间匹配。

        Args:
            dts: list of datetime
            tolerance_hours: 覆盖默认容差

        Returns:
            (matched_indices, valid_mask)
            matched_indices: np.ndarray of int, 与 dts 等长（无效位置填 -1）
            valid_mask: np.ndarray of bool
        """
        tol = (tolerance_hours or self.tolerance_hours) * 3600.0
        epoch = datetime(1970, 1, 1)
        target_hours = np.array(
            [(dt - epoch).total_seconds() / 3600.0 for dt in dts], dtype=np.float64
        )

        # 对每个 target 找最近的 self._times_hours
        # 用广播计算差值（target 较小时合理）
        diffs = np.abs(target_hours[:, None] - self._times_hours[None, :])
        min_indices = np.argmin(diffs, axis=1)
        min_diffs = diffs[np.arange(len(dts)), min_indices]

        valid_mask = min_diffs <= tol
        matched_indices = np.where(valid_mask, min_indices, -1).astype(np.int64)
        return matched_indices, valid_mask

    def get_sst(self, time_indices):
        """按时间索引批量读取 SST。

        Args:
            time_indices: list/array of int（来自 nearest_time_index/nearest_time_indices）

        Returns:
            np.ndarray of shape (n_indices, n_lat, n_lon), float32
        """
        n_lat, n_lon = len(self.lat), len(self.lon)
        result = np.full((len(time_indices), n_lat, n_lon), np.nan, dtype=np.float32)

        # 按文件分组读取，减少 I/O
        file_groups = {}
        for out_idx, glb_idx in enumerate(time_indices):
            fidx, local_idx = self._file_indices[glb_idx]
            file_groups.setdefault(fidx, []).append((out_idx, local_idx))

        for fpath_idx, reads in file_groups.items():
            fpath = self.files[fpath_idx]
            if fpath not in self._sst_cache:
                with NCDataset(fpath, "r") as ds:
                    thetao = ds.variables["thetao"][:, 0, :, :]  # squeeze depth
                    self._sst_cache[fpath] = np.array(thetao, dtype=np.float32)
            cached = self._sst_cache[fpath]
            for out_idx, local_idx in reads:
                result[out_idx] = cached[local_idx]

        return result
