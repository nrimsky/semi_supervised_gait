DOWN_SAMPLE = 5
WINDOW_SIZE = 40
STEP = 2
AUGMENT_STD = 8
X_COLS_LEFT = ("acc_x_left", "acc_y_left", "acc_z_left", "gyr_x_left", "gyr_y_left", "gyr_z_left")
X_COLS_RIGHT = ("acc_x_right", "acc_y_right", "acc_z_right", "gyr_x_right", "gyr_y_right", "gyr_z_right")
SWAP_COLS = ("acc_y_left", "gyr_x_left", "gyr_z_left", "acc_y_right", "gyr_x_right", "gyr_z_right")
Y_COL_LEFT = "left_phase"
Y_COL_RIGHT = "right_phase"
EMBED_DIM = 4
PROJECTION_DIM = 100
SIMCLR_HIDDEN_DIM = 100
N_CHANNELS_CONV = (12, 8, 16)
N_CHANNELS_CONV_LG = (12, 16, 32)
POOL_SIZE = (2, 1)
CONV_SIZE = (5, 1)
RAINBOW = [
    "#ff1e00",
    "#ff8000",
    "#ffe600",
    "#aaff00",
    "#33ff00",
    "#00ff66",
    "#00fffb",
    "#008cff",
    "#0022ff",
    "#6a00ff",
    "#c300ff",
    "#ff00cc",
    "#000000"
]
NORMAL_LR = 1e-3
MPL_LR = 1e-3
SIMCLR_LR = 1e-3
SIMCLR_FINETUNE_EPOCHS = 20
N_EPOCHS_BASELINE = 15
N_EPOCHS = 15
BATCH_SIZE = 128
FINE_TUNE_LR = 1e-4
GRL_ALPHA = 0.2
N_EPOCHS_FINETUNE = 10
PSEUDO_LABELLING_THRESHOLD = 0.17
FRACTION_UNLABELLED = 0.7
FRACTION_TEST = 0.1
