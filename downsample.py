"""
MaCOM → GLO12 降采样算子。
使用 bilinear interpolation 将高分辨率 patch 降采样到 GLO12 网格。
"""

import numpy as np
import torch
import torch.nn.functional as F


def build_downsample_grid(
    src_lat, src_lon, src_h, src_w,
    tgt_lat, tgt_lon, tgt_h, tgt_w,
    y_offset=0, x_offset=0,
    patch_h=None, patch_w=None,
):
    """构建 grid_sample 所需的归一化坐标网格。

    Args:
        src_lat, src_lon: 源网格（MaCOM）的 1D 坐标
        src_h, src_w: 源网格完整尺寸
        tgt_lat, tgt_lon: 目标网格（GLO12）的 1D 坐标
        tgt_h, tgt_w: 目标网格尺寸
        y_offset, x_offset: 当前 patch 在完整网格中的偏移

    Returns:
        grid: (1, tgt_h, tgt_w, 2) 归一化坐标，范围 [-1, 1]
    """
    lat_step = (src_lat[-1] - src_lat[0]) / (src_h - 1)
    lon_step = (src_lon[-1] - src_lon[0]) / (src_w - 1)

    # GLO12 格点在完整网格中的浮点索引
    tgt_y = (tgt_lat - src_lat[0]) / lat_step
    tgt_x = (tgt_lon - src_lon[0]) / lon_step

    # 转换到 patch 内的局部坐标
    tgt_y = tgt_y - y_offset
    tgt_x = tgt_x - x_offset

    # 归一化到 [-1, 1]（grid_sample 的坐标系）
    if patch_h is None:
        patch_h = src_h
    if patch_w is None:
        patch_w = src_w
    norm_y = 2.0 * tgt_y / (patch_h - 1) - 1.0
    norm_x = 2.0 * tgt_x / (patch_w - 1) - 1.0

    grid_x, grid_y = np.meshgrid(norm_x, norm_y)
    grid = np.stack([grid_x, grid_y], axis=-1)

    return torch.from_numpy(grid).float().unsqueeze(0)


def downsample_to_glo12(data, grid, mode="bilinear", padding_mode="zeros", mask=None, eps=1e-6):
    """将高分辨率数据降采样到 GLO12 网格。

    当提供 mask 时，使用归一化平均避免陆地 / NaN 像素的污染。

    Args:
        data: (batch, channels, src_h, src_w) 或 (channels, src_h, src_w)
        grid: (1, tgt_h, tgt_w, 2)，来自 build_downsample_grid
        mode: "bilinear" 或 "nearest"
        padding_mode: "zeros"（超出范围=0）或 "border"（复制边缘）
        mask: 可选 (H, W) 或 (1, 1, H, W) 浮点 mask（1=有效，0=无效）

    Returns:
        (batch, channels, tgt_h, tgt_w) 或 (channels, tgt_h, tgt_w)
    """
    squeeze = False
    if data.dim() == 3:
        data = data.unsqueeze(0)
        squeeze = True

    batch, channels, src_h, src_w = data.shape
    tgt_h, tgt_w = grid.shape[1], grid.shape[2]

    # grid_sample 需要 (N, C, H_in, W_in) 和 (N, H_out, W_out, 2)
    # 但我们的 grid 可能是 (1, tgt_h, tgt_w, 2) 或 (batch, tgt_h, tgt_w, 2)
    if grid.shape[0] == 1 and batch > 1:
        grid_expanded = grid.expand(batch, -1, -1, -1)
    elif grid.shape[0] == batch:
        grid_expanded = grid
    else:
        raise ValueError(f"Grid batch size {grid.shape[0]} does not match data batch size {batch}")

    if mask is None:
        # 对每个 channel 分别做 grid_sample（避免显存问题）
        # 或者一次性处理所有 channels
        result = F.grid_sample(
            data, grid_expanded,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,
        )
    else:
        mask_t = mask
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)
        elif mask_t.dim() == 3:
            mask_t = mask_t.unsqueeze(1)
        elif mask_t.dim() != 4:
            raise ValueError(f"Mask dim {mask_t.dim()} is not supported")

        mask_t = mask_t.float()
        if mask_t.shape[0] == 1 and batch > 1:
            mask_t = mask_t.expand(batch, -1, -1, -1)
        elif mask_t.shape[0] != batch:
            raise ValueError(f"Mask batch size {mask_t.shape[0]} does not match data batch size {batch}")

        num = F.grid_sample(
            data * mask_t, grid_expanded,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,
        )
        den = F.grid_sample(
            mask_t, grid_expanded,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        )
        result = torch.where(den > eps, num / (den + eps), torch.zeros_like(num))

    if squeeze:
        result = result.squeeze(0)

    return result


def downsample_mask_nearest(mask_np, src_lat, src_lon, tgt_lat, tgt_lon,
                            y_offset=0, x_offset=0, patch_h=None, patch_w=None):
    """用 nearest-neighbor 将高分辨率二值 mask 下采样到 GLO12 网格。

    最近邻插值避免了 bilinear 在海岸线处产生的混合值。

    Args:
        mask_np: numpy (H, W) 二值 mask（1=海洋，0=陆地）
        src_lat, src_lon: 源网格 1D 坐标
        tgt_lat, tgt_lon: 目标网格 1D 坐标
        y_offset, x_offset: patch 偏移
        patch_h, patch_w: patch 尺寸（默认全图）

    Returns:
        numpy (tgt_h, tgt_w) 二值 mask
    """
    src_h, src_w = mask_np.shape
    if patch_h is None:
        patch_h = src_h
    if patch_w is None:
        patch_w = src_w

    lat_step = (src_lat[-1] - src_lat[0]) / (src_h - 1)
    lon_step = (src_lon[-1] - src_lon[0]) / (src_w - 1)

    tgt_y = np.round((tgt_lat - src_lat[0]) / lat_step - y_offset).astype(int)
    tgt_x = np.round((tgt_lon - src_lon[0]) / lon_step - x_offset).astype(int)

    valid_y = (tgt_y >= 0) & (tgt_y < patch_h)
    valid_x = (tgt_x >= 0) & (tgt_x < patch_w)

    # Clip 用于安全索引（超出范围直接置 0）
    tgt_y = np.clip(tgt_y, 0, patch_h - 1)
    tgt_x = np.clip(tgt_x, 0, patch_w - 1)

    # 2D 索引
    yy, xx = np.meshgrid(tgt_y, tgt_x, indexing='ij')  # (tgt_h, tgt_w)
    out = mask_np[yy, xx].astype(np.float32)
    valid = valid_y[:, None] & valid_x[None, :]
    out[~valid] = 0.0
    return out
