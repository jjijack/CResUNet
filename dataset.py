import torch
import numpy as np
from torch.utils.data import Dataset
from netCDF4 import Dataset as NCDataset

class NCCorrectionDataset(Dataset):
    def __init__(self, forecast_path, reanalysis_path, max_step=120):
        self.forecast_path = forecast_path
        self.reanalysis_path = reanalysis_path
        self.max_step = max_step
        
        # 1. 扫描所有 Runs，不再剔除，而是记录每个 Run 的有效时长
        self.run_info = self._scan_runs()
        print(f"总样本数: {len(self.run_info)}")

    def _scan_runs(self):
        run_info = []
        with NCDataset(self.forecast_path, 'r') as fc_ds, \
             NCDataset(self.reanalysis_path, 'r') as ra_ds:
            
            total_runs = fc_ds.dimensions['run'].size
            fc_valid_times = fc_ds.variables['valid_time'][:] 
            ra_times = ra_ds.variables['time'][:]
            
            # 建立真值时间索引
            ra_time_map = {round(t, 2): i for i, t in enumerate(ra_times)}
            
            for run_idx in range(total_runs):
                steps_times = fc_valid_times[run_idx]
                
                # 找出哪些 step 有对应的真值
                valid_indices = [] # 记录真值文件里的索引
                valid_steps_count = 0
                
                for t in steps_times:
                    t_round = round(t, 2)
                    if t_round in ra_time_map:
                        valid_indices.append(ra_time_map[t_round])
                        valid_steps_count += 1
                    else:
                        # 一旦超出真值范围，后面通常也都没有了
                        break 
                
                # 只要至少有1个小时是重合的，就算有效样本
                if valid_steps_count > 0:
                    run_info.append({
                        "run_idx": run_idx,
                        "ra_indices": valid_indices,
                        "valid_len": valid_steps_count
                    })
                    
        return run_info

    def __len__(self):
        return len(self.run_info)

    def __getitem__(self, idx):
        info = self.run_info[idx]
        run_idx = info['run_idx']
        ra_indices = info['ra_indices']
        valid_len = info['valid_len']
        
        with NCDataset(self.forecast_path, 'r') as fc_ds, \
             NCDataset(self.reanalysis_path, 'r') as ra_ds:
            
            # 1. 读取完整预报 (Input)
            # 形状 [120, H, W]
            x_raw = fc_ds.variables['sst'][run_idx, :, :, :]
            
            # 2. 读取部分真值 (Target)
            # 形状 [valid_len, H, W]
            y_partial = ra_ds.variables['sst'][ra_indices, :, :]
            
            # 3. 构造 Padding 后的真值和 Mask
            # 初始化全0矩阵 [120, H, W]
            y_full = np.zeros_like(x_raw)
            mask = np.zeros_like(x_raw)
            
            # 填入有效部分
            y_full[:valid_len, :, :] = y_partial
            mask[:valid_len, :, :] = 1.0  # 有效部分设为 1
            
            # 处理 NaN
            x_data = np.nan_to_num(x_raw, nan=0.0)
            y_data = np.nan_to_num(y_full, nan=0.0)
            
            return (torch.from_numpy(x_data).float(), 
                    torch.from_numpy(y_data).float(), 
                    torch.from_numpy(mask).float())