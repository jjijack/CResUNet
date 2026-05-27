experiment_params = {
    "normalize_flag": True,       # 是否进行数据归一化 (建议开启)
    "model": "CResU_Net",
    "device": 'cuda',
    "operation_mode": "train_mode",
    "save_dir": './train_results_macom',  # 结果保存路径
    "model_save_name": 'best_model.pth'   # 最佳模型文件名
}

data_params = {
    # 选择数据管线: "macom"(高分辨率 MaCOM + GLO12 低分辨率监督)
    "mode": "macom",
    "macom": {
        "forecast_pattern": '/home/user3/scratch/SST Correction/CURRENT_MACOM_SH/*/*_swt_SH_001h_*.nc',
        "glo12_pattern": '/home/user3/scratch/SST Correction/GLO12/*.nc',
        "time_tolerance_hours": 0.6,    # GLO12 时间匹配容差（小时）
        "target_h": 1168,               # MaCOM 网格高度
        "target_w": 1000,               # MaCOM 网格宽度
        "target_steps": 168,            # 预报时效（小时）
    },
    # 旧版 FVCOM 管线（保留兼容）
    "fvcom": {
        "forecast_path": './data/forecast_structured.nc',
        "reanalysis_path": './data/reanalysis_structured.nc',
    },

    # 数据集按月切分
    "monthly_split": {
        "train_days": 25,      # 每月前 25 天训练
        "val_days": 3,         # 接着 3 天验证
        "test_days": 2         # 最后 2 天测试
    },

    "num_workers": 4,          # DataLoader 线程数
    "check_files": False,      # 是否检查文件完整性
}

model_params = {
    "CResU_Net": {
        "batch_gen": {
            "batch_size": 1,      # 全图训练 batch=1
            "shuffle": True,
            "seed": 42            # 固定随机种子
        },
        "trainer": {
            "num_epochs": 500,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "optimizer": "adam",

            # 学习率调度器: CosineAnnealingWarmRestarts（周期性重启，帮助跳出局部最优）
            "lr_scheduler": {
                "T_0": 10,         # 初始重启周期（epoch）
                "T_mult": 2,       # 每次重启周期翻倍
                "eta_min": 1e-6,   # 最小学习率
            },

            # 训练过程可视化
            "vis_patch_samples": False,
            "vis_full_domain": True,  # 每次性能提升时生成全图对比图

            # 早停机制
            "early_stopping": {
                "tolerance": 20,       # 连续 N 轮无提升即停止
                "delta": 1e-6          # 最小下降幅度
            },

            # Loss 权重
            "loss_weights": {
                "enabled": True,
                "start_weight": 1.0,   # T=0 的权重
                "end_weight": 1.0,     # T=167 的权重（等权）
                "ignore_steps": 0,     # 前 N 小时不计入损失
                "ignore_weight": 0.0,  # 被忽略时间段的权重
                "tv_weight": 0.05,     # 平滑约束 (TV Loss) 权重
                "l1_weight": 0.05,     # 稀疏性惩罚，保护 0 偏差区域
                "var_weight": 0.0,     # 空间方差 loss 权重（0=关闭）
            },
        },
        "core": {
            "in_channels": 171,   # 168 SST + 1 land mask + 2 position encoding (lat, lon)
            "out_channels": 168,  # 输出 168 时间步的偏差场
            "base_channels": 64,  # 基础通道数（控制模型宽度）
            "dropout": 0.1,       # 瓶颈层随机丢弃率
        }
    },
}
