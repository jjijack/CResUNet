
experiment_params = {
    "normalize_flag": True,
    "model": "CResU_Net",
    "device": 'cuda',
    "operation_mode": "train_mode"
}

data_params = {
    "weather_raw_dir": 'F:\\OSTIA',
    "area_name": "NSCS",
    "spatial_range": [[17, 22], [113, 120]],  # NSCS [17, 22], [113, 120], WSCS[8, 18], [109, 113], ESCS[12, 16],[115, 120]
    "weather_freq": 1,
    "downsample_mode": "selective",  # can be average or selective
    "check_files": False,
    "features": ['analysed_sst'],
    "target_dim": 0,
    "rebuild": False,

    "train_period": {
            "start": "1995-01-01",
            "end": "2010-12-31"
    },
    "val_period": {
            "start": "2011-01-01",
            "end": "2012-12-31"
    },
    "test_period": {
            "start": "2013-01-01",
            "end": "2013-12-31"
    }
}

model_params = {
    "CResU_Net": {
        "batch_gen": {
            "input_dim": 0,
            "output_dim": 0,
            "window_in_len": 10,
            "window_out_len": 10,
            "batch_size": 2,
            "shuffle": True,
            "seed": 1
        },
        "trainer": {
            "num_epochs": 1,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.0002,
            "learning_rate": 0.0001,
            "clip": 5,
            "early_stop_tolerance": 8
        },
        "core": {
            "selected_dim": 0,
            "in_channels": 121,
            "out_channels": 120
        }
    },
}
