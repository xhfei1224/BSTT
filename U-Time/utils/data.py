import glob
import os
from typing import List
from functools import partial

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from utils.file_loader import load_npz_files


class SleepData(Dataset):
    def __getitem__(self, idx) -> T_co:
        return self.data[idx, ...], self.labels[idx]

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data, self.labels = data, labels

    def __len__(self):
        return self.data.shape[0]

    def normalization(self):
        self.data -= np.mean(self.data)
        self.data /= np.std(self.data)


def _get_datasets(modals: int, data_dir: str, two_d: bool = True) -> List[SleepData]:
    data, labels = load_npz_files(glob.glob(os.path.join(data_dir, '*.npz')), two_d=two_d)
    if modals is None:
        datasets = [SleepData(d, l) for d, l in zip(data, labels)]
    else:
        datasets = [SleepData(d[..., modals], l) for d, l in zip(data, labels)]
    return datasets


get_eeg_datasets = partial(_get_datasets, 0)
get_eog_datasets = partial(_get_datasets, 1)
get_emg_datasets = partial(_get_datasets, 2)
get_datasets = partial(_get_datasets, None)
