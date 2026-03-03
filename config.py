experiment_params = {
    "normalize_flag": True,       # 是否进行数据归一化 (建议开启)
    "model": "CResU_Net",
    "device": 'cuda',
    "operation_mode": "train_mode",
    "save_dir": './train_results',      # 结果保存路径
    "model_save_name": 'best_model.pth' # 最佳模型文件名
}

data_params = {
    # .nc 文件路径
    "forecast_path": './data/forecast_structured.nc',
    "reanalysis_path": './data/reanalysis_structured.nc',
    
    # 数据集切分参数 (替代原本的日期切分)
    "monthly_split": {
        "train_days": 20,      # 每月前 20 天训练
        "val_days": 5,         # 接着 5 天验证
        "test_days": 5         # 最后 5 天测试
    },

    "num_workers": 4,          # DataLoader 线程数
    "check_files": False,      # 是否检查文件完整性
    
    # --- 原始模型的参数，在当前 .nc 模式下可能暂时用不到，但保留以防万一 ---
    "weather_raw_dir": 'F:\\OSTIA', 
    "area_name": "NSCS",
    "spatial_range": [[17, 22], [113, 120]], 
    "downsample_mode": "selective",
    "features": ['analysed_sst'],
}

model_params = {
    "CResU_Net": {
        "batch_gen": {
            "batch_size": 2,      # 显存允许的话可以设为 4 或 8
            "shuffle": True,
            "seed": 42            # 固定随机种子
        },
        "trainer": {
            "num_epochs": 500,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,       # 稍微减小正则化力度
            "optimizer": "adam",
            
            # --- 学习率调度器 (ReduceLROnPlateau) ---
            "lr_scheduler": {
                "mode": 'min',
                "factor": 0.5,         # 每次衰减一半
                "patience": 5,         # 忍受5轮不下降
            },
            
            # --- 早停机制 ---
            "early_stopping": {
                "tolerance": 20,       # 忍受15轮 (原 early_stop_tolerance)
                "delta": 1e-6          # 最小下降幅度
            },

            # --- 时间加权 Loss ---
            "loss_weights": {
                "enabled": True,       # 是否开启加权
                "start_weight": 1.0,   # T=0 的权重
                "end_weight": 10.0,    # T=119 的权重
                "ignore_steps": 25,    # 前 N 小时不计入损失 (例如 0-24 共25小时)
                "ignore_weight": 0.0,  # 被忽略时间段的权重
                "tv_weight": 0.01,     # 平滑约束(TV Loss)权重，建议 0.005 ~ 0.02
                "l1_weight": 0.1       # 稀疏性惩罚，保护 0 偏差区域
            }
        },
        "core": {
            "in_channels": 121,        # 120 SST + 1 Mask
            "out_channels": 120,       # 输出必须是 120 (预测每个时间步的 Bias)
        }
    },
}