from utils.data import SleepData
from functools import partial
from utils.file_loader import load_npz_files
import os
import glob
from typing import List
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def _get_datasets(modals: int, data_dir: str, stride: int = 35, two_d: bool = True) -> List[SleepData]:
    data, labels = load_npz_files(glob.glob(os.path.join(data_dir, '*.npz')), two_d=two_d)

    def data_big_group(d: np.ndarray) -> np.ndarray:
        """
        A closure to divide data into big groups to prevent data leak in data enhancement
        """
        return_data = np.array([])
        beg = 0
        while (beg + stride) <= d.shape[0]:
            y = d[beg: beg + stride, ...]
            y = y.reshape((1, 1, 35, 3000, 3))
            # y = y[np.newaxis, ...]
            return_data = y if beg == 0 else np.append(return_data, y, axis=0)
            beg += stride
        return return_data

    def label_big_group(l: np.ndarray) -> np.ndarray:
        """
        A closure to divide labels into big groups to prevent data leak in data enhancement
        """
        return_labels = np.array([])
        beg = 0
        while (beg + stride) <= len(l):
            y = l[beg: beg + stride]
            y = y[np.newaxis, ...]
            return_labels = y if beg == 0 else np.concatenate((return_labels, y), axis=0)
            beg += stride
        return return_labels  # [:, np.newaxis, ...]

    with ThreadPoolExecutor(max_workers=4) as executor:
        data = executor.map(data_big_group, data)
        labels = executor.map(label_big_group, labels)

    if modals is None:
        datasets = [SleepData(d, l) for d, l in zip(data, labels)]
    else:
        datasets = [SleepData(d[..., modals], l) for d, l in zip(data, labels)]
    return datasets


get_eeg_datasets = partial(_get_datasets, 0)
get_eog_datasets = partial(_get_datasets, 1)
get_emg_datasets = partial(_get_datasets, 2)
get_datasets = partial(_get_datasets, None)


if __name__ == '__main__':
    import time
    start = time.time()
    a = get_eeg_datasets('../../sleep_data/sleepedf-39')
    print(time.time() - start)
