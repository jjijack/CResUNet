"""
从 out_macom 订正 SST 产品生成动态 GIF。
用法：
    python make_gif.py                             # 单文件 168h 动图（每6h一帧）
    python make_gif.py --all-files                 # 多文件逐日起报 T+0h 对比动图
    python make_gif.py --nc out_macom/xxx.nc       # 指定文件
    python make_gif.py --step 3                    # 每 3h 一帧
"""

import argparse
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from netCDF4 import Dataset as NCDataset, num2date
from PIL import Image
import io

matplotlib.rcParams["axes.unicode_minus"] = False

# ─── 绘图参数 ─────────────────────────────────────────
CMAP       = "jet"
FIGSIZE    = (7, 6)
DPI        = 100
INTERVAL   = 250   # ms / 帧 (GIF 帧间隔)
FONT_SIZE  = 11


def load_nc(path):
    with NCDataset(path) as ds:
        sst  = ds.variables["sst"][:]          # (T, H, W)
        lat  = ds.variables["lat"][:]
        lon  = ds.variables["lon"][:]
        tvar = ds.variables["time"]
        times = num2date(tvar[:], units=tvar.units,
                         calendar=getattr(tvar, "calendar", "standard"))
    return sst, lat, lon, times


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img.copy()


def make_single_file_gif(nc_path, step=6, out_dir=".", label="corrected"):
    """单文件 168h 动图：逐时次订正 SST 演变。"""
    sst, lat, lon, times = load_nc(nc_path)
    n_steps = sst.shape[0]

    # 全局色标
    ocean = sst[~np.isnan(sst)]
    vmin, vmax = float(np.percentile(ocean, 2)), float(np.percentile(ocean, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP)
    cmap.set_bad("lightgray")

    basename = os.path.splitext(os.path.basename(nc_path))[0]
    gif_path = os.path.join(out_dir, f"{basename}_{label}.gif")

    frames = []
    indices = range(0, n_steps, step)
    print(f"生成 {len(list(indices))} 帧 (step={step}h) → {gif_path}")

    for t in indices:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        im = ax.pcolormesh(lon, lat, sst[t], cmap=cmap, norm=norm, shading="auto")
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
        cb.set_label("SST (°C)", fontsize=FONT_SIZE)
        ax.set_xlabel("Longitude (°E)", fontsize=FONT_SIZE)
        ax.set_ylabel("Latitude (°N)", fontsize=FONT_SIZE)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        dt = times[t]
        ax.set_title(
            f"AI-Corrected SST  |  Init: {times[0].strftime('%Y-%m-%d %H:00 UTC')}"
            f"\nT+{t:3d}h  ({dt.strftime('%Y-%m-%d %H:00 UTC')})",
            fontsize=FONT_SIZE,
        )
        fig.tight_layout()
        frames.append(fig_to_pil(fig))
        plt.close(fig)

    # 循环一次，最后重复 4 帧结尾停顿
    frames += [frames[-1]] * 4
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=INTERVAL,
        loop=0,
        optimize=True,
    )
    print(f"✅ 已保存: {gif_path}")
    return gif_path


def make_multifile_gif(nc_dir, step=24, out_dir=".", label="T0_daily"):
    """多文件动图：每个起报文件取 T+0h 帧，展示逐日订正 SST。"""
    files = sorted(glob.glob(os.path.join(nc_dir, "AI_CResU-net_sstR_SH_*.nc")))
    if not files:
        raise FileNotFoundError(f"No NC files in {nc_dir}")
    print(f"共 {len(files)} 个起报文件")

    # 先扫描全局色标
    all_vals = []
    for f in files:
        sst, _, _, _ = load_nc(f)
        all_vals.append(sst[0][~np.isnan(sst[0])].ravel())
    all_vals = np.concatenate(all_vals)
    vmin, vmax = float(np.percentile(all_vals, 2)), float(np.percentile(all_vals, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP)
    cmap.set_bad("lightgray")

    _, lat, lon, _ = load_nc(files[0])
    gif_path = os.path.join(out_dir, f"macom_corrected_{label}.gif")
    frames = []

    for f in files:
        sst, _, _, times = load_nc(f)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        im = ax.pcolormesh(lon, lat, sst[0], cmap=cmap, norm=norm, shading="auto")
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
        cb.set_label("SST (°C)", fontsize=FONT_SIZE)
        ax.set_xlabel("Longitude (°E)", fontsize=FONT_SIZE)
        ax.set_ylabel("Latitude (°N)", fontsize=FONT_SIZE)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_title(
            f"AI-Corrected SST  T+0h  |  Init: {times[0].strftime('%Y-%m-%d %H:00 UTC')}",
            fontsize=FONT_SIZE,
        )
        fig.tight_layout()
        frames.append(fig_to_pil(fig))
        plt.close(fig)
        print(f"  {os.path.basename(f)} → frame added")

    frames += [frames[-1]] * 4
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=INTERVAL * 2,
        loop=0,
        optimize=True,
    )
    print(f"✅ 已保存: {gif_path}")
    return gif_path


def load_forecast_nc(path):
    """读取原始 MaCOM 预报文件（变量名 't'，shape=(T,1,H,W)）。"""
    with NCDataset(path) as ds:
        raw = ds.variables["t"][:, 0, :, :]   # squeeze depth dim → (T, H, W)
        sst = np.array(raw, dtype=np.float32)
        if np.ma.is_masked(raw):
            sst[raw.mask] = np.nan
        lat  = ds.variables["lat"][:]
        lon  = ds.variables["lon"][:]
        tvar = ds.variables["time"]
        # 原始文件 units="seconds"，实际单位存在 long_name 里
        time_units = tvar.units
        if "since" not in time_units:
            long_name = getattr(tvar, "long_name", "")
            time_units = long_name if "since" in long_name else "seconds since 1949-10-01 00:00:00"
        times = num2date(tvar[:], units=time_units,
                         calendar=getattr(tvar, "calendar", "standard"))
    return sst, lat, lon, times


def make_forecast_gif(nc_path, step=6, out_dir=".", label="forecast"):
    """原始 MaCOM 预报场动图（168h 演变）。"""
    sst, lat, lon, times = load_forecast_nc(nc_path)
    n_steps = sst.shape[0]

    ocean = sst[~np.isnan(sst)]
    vmin, vmax = float(np.percentile(ocean, 2)), float(np.percentile(ocean, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP)
    cmap.set_bad("lightgray")

    basename = os.path.splitext(os.path.basename(nc_path))[0]
    gif_path = os.path.join(out_dir, f"{basename}_{label}.gif")

    frames = []
    indices = range(0, n_steps, step)
    print(f"生成 {len(list(indices))} 帧 (step={step}h) → {gif_path}")

    for t in indices:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        im = ax.pcolormesh(lon, lat, sst[t], cmap=cmap, norm=norm, shading="auto")
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
        cb.set_label("SST (°C)", fontsize=FONT_SIZE)
        ax.set_xlabel("Longitude (°E)", fontsize=FONT_SIZE)
        ax.set_ylabel("Latitude (°N)", fontsize=FONT_SIZE)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        dt = times[t]
        ax.set_title(
            f"MaCOM Forecast SST  |  Init: {times[0].strftime('%Y-%m-%d %H:00 UTC')}"
            f"\nT+{t:3d}h  ({dt.strftime('%Y-%m-%d %H:00 UTC')})",
            fontsize=FONT_SIZE,
        )
        fig.tight_layout()
        frames.append(fig_to_pil(fig))
        plt.close(fig)

    frames += [frames[-1]] * 4
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=INTERVAL,
        loop=0,
        optimize=True,
    )
    print(f"✅ 已保存: {gif_path}")
    return gif_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成订正/预报 SST GIF 动图")
    parser.add_argument("--nc",        default=None,
                        help="指定单个订正 NC 文件（默认取 out_macom/ 第一个）")
    parser.add_argument("--forecast",  default=None,
                        help="原始 MaCOM 预报 NC 文件（变量名 t）")
    parser.add_argument("--nc-dir",    default="out_macom",
                        help="订正结果目录（default: out_macom）")
    parser.add_argument("--out-dir",   default="docs",
                        help="GIF 输出目录（default: docs）")
    parser.add_argument("--step",      type=int, default=6,
                        help="帧间隔（小时，default: 6）")
    parser.add_argument("--all-files", action="store_true",
                        help="多文件逐日动图（每文件取 T+0h）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.all_files:
        make_multifile_gif(args.nc_dir, out_dir=args.out_dir)
    elif args.forecast:
        make_forecast_gif(args.forecast, step=args.step, out_dir=args.out_dir)
    else:
        if args.nc:
            nc_path = args.nc
        else:
            files = sorted(glob.glob(os.path.join(args.nc_dir, "*.nc")))
            if not files:
                raise FileNotFoundError(f"No NC files in {args.nc_dir}")
            nc_path = files[0]
            print(f"使用文件: {nc_path}")
        make_single_file_gif(nc_path, step=args.step, out_dir=args.out_dir)
