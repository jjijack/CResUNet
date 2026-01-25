import torch
import numpy as np
from torch.utils.data import Dataset
from netCDF4 import Dataset as NCDataset

class NCCorrectionDataset(Dataset):
    def __init__(self, forecast_path, reanalysis_path):
        self.forecast_path = forecast_path
        self.reanalysis_path = reanalysis_path
        
        print("正在初始化数据集索引 ...")
        
        # --- 1. 预读取轻量级元数据 ---
        with NCDataset(self.forecast_path, 'r') as fc_ds:
            self.total_runs = fc_ds.dimensions['run'].size
            # 仅读取时间，不读 SST
            self.fc_times = fc_ds.variables['valid_time'][:] 
            
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
        print(f"索引构建完成！有效样本: {len(self.run_info)}")

    def __len__(self):
        return len(self.run_info)

    def __getitem__(self, idx):
        # 获取索引信息
        info = self.run_info[idx]
        run_idx = info['run_idx']
        ra_indices = info['ra_indices']
        valid_len = info['valid_len']
        
        # 1. 读取 Forecast SST
        with NCDataset(self.forecast_path, 'r') as fc_ds:
            sst_raw = fc_ds.variables['sst'][run_idx] 
            sst_input = np.nan_to_num(sst_raw, nan=0.0)
            
        # 2. 拼接 Land Mask
        x_combined = np.concatenate((sst_input, self.land_mask_cache), axis=0)
        
        # 3. 读取 Reanalysis (Target)
        y_full = np.zeros_like(sst_input)
        
        # --- 【核心修改】构造复合 Mask (时间 + 空间) ---
        # 初始化 Mask：全 0
        combined_mask = np.zeros_like(sst_input)
        
        if valid_len > 0:
            with NCDataset(self.reanalysis_path, 'r') as ra_ds:
                y_partial = ra_ds.variables['sst'][ra_indices]
                y_partial = np.nan_to_num(y_partial, nan=0.0)
                
            y_full[:valid_len] = y_partial
            
            # 只有在 (有效时间) AND (是海洋) 的地方，Mask 才为 1
            # 利用广播机制：time_mask (前N层为1) * ocean_mask (空间为1)
            combined_mask[:valid_len] = self.land_mask_cache
        return (torch.from_numpy(x_combined).float(), 
                torch.from_numpy(y_full).float(), 
                torch.from_numpy(combined_mask).float()) # 返回复合 Mask