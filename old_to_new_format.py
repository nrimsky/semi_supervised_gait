"""
Script to process legacy CSV files into new format
"""

import glob
import pandas as pd
from globals import X_COLS_LEFT, X_COLS_RIGHT
import numpy as np
from helpers import ensure_dir
import os


def split_cycle(y, n_splits):
    try:
        new_vec = y.numpy()
    except:
        new_vec = y.copy()
    try:
        start_inds = np.where(np.concatenate(([y[0]], y[:-1] != y[1:], [99])))[0]
        # Get starts and durations for stance and swing
        phase_durs = np.diff(start_inds)
        incomplete_head = phase_durs[0]
        incomplete_tail = phase_durs[-1] or -len(y)  # prevent zero returns: eliminates the entire vector because we do data[head: -tail]
    except IndexError:
        print('Insufficient phase data found')
        return y, 0, -len(y)
    if n_splits == 0: return y, incomplete_head, incomplete_tail
    phase_durs = phase_durs[1:-1]
    start_inds = start_inds[1:-1]
    swing_start_inds = [i for i in start_inds if (y[i] == 0)]
    stance_start_inds = [i for i in start_inds if (y[i] == 1)]
    try:
        start_in_swing = (new_vec[start_inds[0]] == 0).astype(int)  # test if we start in swing
    except IndexError:  # No swing phases present - may happen in testing
        return y, incomplete_head, incomplete_tail
    start_in_stance = np.logical_not(start_in_swing).astype(int)
    stance_durs = phase_durs[start_in_swing::2]
    swing_durs = phase_durs[start_in_stance::2]
    for i, j in zip(swing_start_inds, swing_durs[:]):
        for t in range(n_splits):
            k = i + t * j // n_splits
            f = i + (t + 1) * j // n_splits
            new_vec[k:f] = t
    prev_dur = 0
    for i, j in zip(stance_start_inds, stance_durs[:]):
        # deal with long standstills. later part is counted as its own label. up to this point, assign
        # labels up to mid-stance, linearly, to account for the fact that it is stance-like!
        if j > 500:
            half_point = prev_dur // 2
            new_vec[i + half_point:i + j] = n_splits * 2
            for t in range(n_splits // 2):
                k = i + t * half_point // (n_splits // 2)
                f = i + (t + 1) * half_point // (n_splits // 2)
                new_vec[k:f] = t + (n_splits)
            continue
        for t in range(n_splits):
            k = i + t * j // n_splits
            f = i + (t + 1) * j // n_splits
            new_vec[k:f] = t + n_splits
        prev_dur = j
    return new_vec, incomplete_head, incomplete_tail


def get_filenames():
    names = glob.glob("old_data/*/*.csv")
    files_to_use = []
    subs = set()
    for n in names:
        if "shank" in n:
            continue
        if "run" in n or "walk" in n or "jog" in n:
            files_to_use.append(n)
            subs.add(os.path.split(n)[1])
    subs = list(subs)
    subs = {t[1]: t[0] for t in enumerate(subs)}
    return files_to_use, subs


def process_data(filename):
    df = pd.read_csv(filename)
    df = df.rename(columns={
        "IMU1_acc_x": "acc_y_left",
        "IMU1_acc_y": "acc_x_left",
        "IMU1_acc_z": "acc_z_left",
        "IMU1_gyr_x": "gyr_y_left",
        "IMU1_gyr_y": "gyr_x_left",
        "IMU1_gyr_z": "gyr_z_left",
        "IMU2_acc_x": "acc_y_right",
        "IMU2_acc_y": "acc_x_right",
        "IMU2_acc_z": "acc_z_right",
        "IMU2_gyr_x": "gyr_y_right",
        "IMU2_gyr_y": "gyr_x_right",
        "IMU2_gyr_z": "gyr_z_right",
        "l_stance": "left_phase",
        "r_stance": "right_phase"
    })
    df["acc_z_left"] *= 1
    df["acc_y_right"] *= 1
    df["acc_z_right"] *= 1
    df["gyr_y_left"] *= 1
    df["gyr_x_left"] *= 1
    df["gyr_y_right"] *= 1
    df["gyr_z_right"] *= 1
    df = df[list(X_COLS_LEFT) + list(X_COLS_RIGHT) + ["left_phase", "right_phase"]]
    incomplete_head = 0
    incomplete_tail = 0
    if np.sum(pd.isnull(df['left_phase'])) == 0:
        df['left_phase'], incomplete_head, incomplete_tail = split_cycle(df['left_phase'].values, 6)
    if np.sum(pd.isnull(df['right_phase'])) == 0:
        df['right_phase'], incomplete_head, incomplete_tail = split_cycle(df['right_phase'].values, 6)
    df = df.iloc[incomplete_head: -incomplete_tail]
    print("Successfully processed data")
    return df


def process_save_data():
    ensure_dir("old_processed_data", empty=True)
    filenames, subjects = get_filenames()
    counts = {s+34: 1 for s in subjects.values()}
    for fn in filenames:
        activity = "walk"
        if "run" in fn:
            activity = "run"
        if "jog" in fn:
            activity = "jog"
        subject = subjects[os.path.split(fn)[1]]+34
        data = process_data(fn)
        if data is not None:
            print(f"Saving data for subject {subject}, activity {activity}, recording {counts[subject]}")
            data.to_csv(f"old_processed_data/0{subject}_{activity}[]_{counts[subject]}_220731_1830", index=False)
            counts[subject] += 1
    print("Done")


if __name__ == '__main__':
    process_save_data()