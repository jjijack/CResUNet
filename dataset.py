import torch
import numpy as np
from torch.utils.data import Dataset
from netCDF4 import Dataset as NCDataset, num2date
from datetime import date

class NCCorrectionDataset(Dataset):
    def __init__(self, forecast_path, reanalysis_path):
        self.forecast_path = forecast_path
        self.reanalysis_path = reanalysis_path
        
        print("正在初始化数据集索引 ...")
        
        # --- 1. 预读取轻量级元数据 ---
        with NCDataset(self.forecast_path, 'r') as fc_ds:
            self.total_runs = fc_ds.dimensions['run'].size
            # 仅读取时间，不读 SST
            time_var = fc_ds.variables['valid_time']
            self.fc_times = time_var[:]
            self.time_units = getattr(time_var, 'units', None)
            self.time_calendar = getattr(time_var, 'calendar', 'standard')
            
            # 【优化】Land Mask 很小，直接读进内存，避免每次反复读硬盘
            # 假设 land_mask 形状是 [lat, lon]，读进来扩展为 [1, H, W]
            raw_mask = fc_ds.variables['land_mask'][:]
            self.land_mask_cache = np.nan_to_num(raw_mask, nan=0.0)[np.newaxis, :, :]

        with NCDataset(self.reanalysis_path, 'r') as ra_ds:
            ra_times = ra_ds.variables['time'][:]
            # 建立时间映射表
            self.ra_time_map = {round(t, 2): i for i, t in enumerate(ra_times)}
            
        # --- 2. 扫描有效 Run ---
        self.run_info = []
        self.run_dates = []
        for run_idx in range(self.total_runs):
            steps_times = self.fc_times[run_idx]
            valid_indices = []
            valid_steps_count = 0
            
            for t in steps_times:
                t_round = round(t, 2)
                if t_round in self.ra_time_map:
                    valid_indices.append(self.ra_time_map[t_round])
                    valid_steps_count += 1
                else:
                    break # 假设时间是连续的，断了就停
            
            if valid_steps_count > 0:
                self.run_info.append({
                    "run_idx": run_idx,
                    "ra_indices": valid_indices, # 存的是索引列表
                    "valid_len": valid_steps_count
                })

                # 记录该 run 的日期 (用于按月切分)
                run_date = None
                if self.time_units:
                    try:
                        dt = num2date(steps_times[0], self.time_units, self.time_calendar)
                        if hasattr(dt, "date"):
                            run_date = dt.date()
                        else:
                            run_date = date(dt.year, dt.month, dt.day)
                    except Exception:
                        run_date = None
                self.run_dates.append(run_date)
        print(f"索引构建完成！有效样本: {len(self.run_info)}")

    def __len__(self):
        return len(self.run_info)

    def get_run_dates(self):
        return self.run_dates

    def __getitem__(self, idx):
        # 获取索引信息
        info = self.run_info[idx]
        run_idx = info['run_idx']
        # 注意：这里我们假设 ra_indices 是一个列表或 range
        ra_indices = info['ra_indices'] 
        
        # 1. 读取 Forecast SST (Input)
        # 假设 Forecast 文件是对齐的，直接读取
        with NCDataset(self.forecast_path, 'r') as fc_ds:
            sst_raw = fc_ds.variables['sst'][run_idx] 
            sst_input = np.nan_to_num(sst_raw, nan=0.0)
            
        # 2. 拼接 Land Mask (Input Channel 120 -> 121)
        x_combined = np.concatenate((sst_input, self.land_mask_cache), axis=0)
        
        # 3. 读取 Reanalysis (Target) & 构造动态 Mask
        # 初始化全 0 (默认假设全是无效的/缺失的)
        y_full = np.zeros_like(sst_input)
        combined_mask = np.zeros_like(sst_input)
        
        with NCDataset(self.reanalysis_path, 'r') as ra_ds:
            # --- 【核心修改】安全读取逻辑 ---
            # 获取 Reanalysis 文件的实际总时间步数
            total_ra_steps = ra_ds.variables['sst'].shape[0]
            
            # 过滤出合法的索引 (必须小于文件总长度)
            # 这样处理后，12月31日的样本只会读到存在的几个小时，超出的会被扔掉
            valid_indices = [i for i in ra_indices if 0 <= i < total_ra_steps]
            real_valid_len = len(valid_indices)
            
            if real_valid_len > 0:
                # 只读取存在的那些时间步
                y_partial = ra_ds.variables['sst'][valid_indices]
                y_partial = np.nan_to_num(y_partial, nan=0.0)
                
                # 填入 y_full 的前 N 个位置
                y_full[:real_valid_len] = y_partial
                
                # --- 构造 Mask ---
                # 只有 (真实存在的时间步) AND (非陆地) 的区域，Mask 才为 1
                # 之后缺失的时间步（比如12月之后的），Mask 保持为 0，Loss 不会计入
                combined_mask[:real_valid_len] = self.land_mask_cache

        # 转 Tensor 并返回
        return (torch.from_numpy(x_combined).float(), 
                torch.from_numpy(y_full).float(), 
                torch.from_numpy(combined_mask).float())