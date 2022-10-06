import argparse
import datetime
import glob
import math
import ntpath
import os
import shutil

import numpy as np
import pandas
from mne.io import read_raw_edf

from edf_header_reader import BaseEDFReader

W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5


stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}


class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}


ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30


def main():
    """This function convert EDF+ files to npz file.
    """
    # Preparing args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str,
                        default="../sleep_data/sleepedf-39",
                        help="File path to the edf file \
                              that contain sleeping info.")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="../sleep_data/sleepedf/prepared",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", '-s', type=list,
                        default=[
                            "EEG Fpz-Cz",
                            "EOG horizontal",
                            "EMG submental"
                        ],
                        help="choose the channels for training.")
    args = parser.parse_args()

    # output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Select channels
    select_ch: list = args.select_ch

    # Read raw and annotation EDF files.
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i, file in enumerate(psg_fnames):
        raw = read_raw_edf(file, preload=True, stim_channel=None)
        sampling_rate = raw.info['sfreq']
        # changed, we choose 3 channels
        raw_ch_df = raw.to_data_frame(scaling_time=100.0)[select_ch]
        if not isinstance(raw_ch_df, pandas.DataFrame):
            raw_ch_df = raw_ch_df.to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))

        # Get raw header
        with open(file, 'r', encoding='iso-8859-1') as f:
            reader_raw = BaseEDFReader(f)
            reader_raw.read_header()
            h_raw = reader_raw.header
        raw_start_dt = datetime.datetime.strptime(
            h_raw['date_time'], "%Y-%m-%d %H:%M:%S"
        )

        # Read annotation and its header
        with open(ann_fnames[i], 'r', encoding='iso-8859-1') as f:
            reader_ann = BaseEDFReader(f)
            reader_ann.read_header()
            h_ann = reader_ann.header
            _, _, ann = list(zip(*reader_ann.records()))
        ann_start_dt = datetime.datetime.strptime(
            h_raw['date_time'], "%Y-%m-%d %H:%M:%S"
        )

        # Assert that raw and annotation files start at the same time
        assert raw_start_dt == ann_start_dt

        # Generate label and remove indices
        remove_idx = []  # indices of the data that will be removed
        labels = []  # indices of the data that have labels
        label_idx = []
        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            label = ann2label[ann_str]
            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Someting wrong")
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=np.int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + \
                    np.arange(duration_sec * sampling_rate, dtype=np.int)
                label_idx.append(idx)

                print(f"Include onset:{onset_sec}, \
                        duration:{duration_sec}, \
                        label:{label}, ({ann_str})")
            else:
                idx = int(onset_sec * sampling_rate) + \
                    np.arange(duration_sec * sampling_rate, dtype=np.int)
                remove_idx.append(idx)

                print(f"Remove onset:{onset_sec}, \
                        duration:{duration_sec}, \
                        label:{label}, ({ann_str})")
        labels = np.hstack(labels)

        print(f'before remove unwanted: {np.arange(len(raw_ch_df)).shape}')
        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        else:
            select_idx = np.arange(len(raw_ch_df))
        print(f"after remove unwanted: {select_idx.shape}")

        # Select only the data with labels
        print(f"before intersect label: {select_idx.shape}")
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)
        print(f"after intersect label: {select_idx.shape}")

        # Remove extra index
        if len(label_idx) > len(select_idx):
            print(f"before remove extra labels: \
                {select_idx.shape}, {labels.shape}")
            extra_idx = np.setdiff1d(label_idx, select_idx)
            # Trim the tail
            if np.all(extra_idx > select_idx[-1]):
                # the original code seems wrong.
                n_label_trims = int(math.ceil(
                    len(extra_idx) / (EPOCH_SEC_SIZE * sampling_rate)
                ))
                if n_label_trims != 0:
                    labels = labels[:-n_label_trims]
            print(f"after remove extra labels: \
                {select_idx.shape}, {labels.shape}")

        # Remove movement and unknown stages if any
        raw_ch = raw_ch_df.values[select_idx]

        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, int(n_epochs))).astype(np.float32)
        y = labels.astype(np.int32)

        assert len(x) == len(y)

        # Select on sleep periods
        w_edge_min = 30
        nw_idx = np.where(y != stage_dict['W'])[0]
        start_idx = nw_idx[0] - (w_edge_min * 2)
        end_idx = nw_idx[-1] + (w_edge_min * 2)
        if start_idx < 0:
            start_idx = 0
        if end_idx >= len(y):
            end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        print(f"Data before selection: {x.shape}, {y.shape}")
        x = x[select_idx]
        y = y[select_idx]
        print(f"Data after selection: {x.shape}, {y.shape}")

        # Save
        filename = ntpath.basename(file).replace("-PSG.edf", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "header_raw": h_raw,
            "header_annotation": h_ann,
        }

        np.savez_compressed(
            os.path.join(args.output_dir, filename),
            **save_dict
        )  # compress size

        print("\n=======================================\n")


if __name__ == '__main__':
    main()
