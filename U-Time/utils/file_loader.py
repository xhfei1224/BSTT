import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor


def load_npz_file(npz_file_name: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load training and validation data from npz files.
    :param npz_file_name: a str of npz filename.
    :return: a tuple of PSG data, labels and sampling rate of the npz file.
    """
    print(f"Loading {npz_file_name}.")
    with np.load(npz_file_name) as f:
        x, y = f['x'], f['y']
        sampling_rate = float(f['fs'])
    return x, y, sampling_rate


# # FIXME: 不能和torch的DataLoader较好兼容，目前暂时输出为[*, 3000, 3, 1, 1]
# def transform_data_axis(partial_jobs: np.ndarray, two_d: bool) -> np.ndarray:
#     """ transform the data axis from [*, 3000, 3] to [3, *, 3000, 1, 1].
#     The extra axis length=1 making data adapted by Conv2d layer.
#     :param two_d: denote data's dimension is adapted by Conv2D else Conv1D.
#     :param partial_jobs: the loaded `x` data.
#     :return: `x` data after axis transform.
#     """
#     partial_jobs = np.squeeze(partial_jobs)  # clear the axis length is 1.
#     if two_d:
#         partial_jobs = partial_jobs[..., np.newaxis, np.newaxis]  # [*, 3000, 3] -> [*, 3000, 3, 1, 1]
#     else:
#         partial_jobs = partial_jobs[..., np.newaxis]  # [*, 3000, 3] -> [*, 3000, 3, 1]
#     # partial_jobs = np.concatenate((partial_jobs[np.newaxis, :, :, 0, :, :], partial_jobs[np.newaxis, :, :, 1, :, :],
#     #                                partial_jobs[np.newaxis, :, :, 2, :, :]),
#     #                               axis=0)  # [*, 3000, 3, 1, 1] -> [3, *, 3000, 1, 1]
#
    # return partial_jobs


def load_npz_files(
        npz_files_name: List[str],
        workers: int = 4,
        two_d: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load data and labels for training and validation.
    Note that we default use 3 channels, if that's changed, need to change the axes stuff code.
    :param two_d: denote data's dimension is adapted by Conv2D else Conv1D.
    :param workers: size of threads pool.
    :param npz_files_name: a list of str contains npz files' name.
    :return: the list of chosen PSG data and labels. Returning with `npz_files_name`'s order.
    """
    assert len(npz_files_name) > 0
    data_list, label_list, fs_list = [], [], []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for record in executor.map(load_npz_file, npz_files_name):
            data_list.append(record[0].astype(np.float32))
            label_list.append(record[1].astype(np.int32))
            fs_list.append(record[2])
            data_list = list(executor.map(lambda x: np.squeeze(x), data_list))
            if two_d:  # Conv2d
                data_list = list(executor.map(lambda x: x[:, np.newaxis, np.newaxis, :, :], data_list))
            else:  # Conv1d
                data_list = list(executor.map(lambda x: x[:, np.newaxis, ...], data_list))

    if len(np.unique(fs_list)) != 1:
        raise Exception("Found mismatch in sampling rate.")

    print(f"load {len(data_list)} files totally.")
    return data_list, label_list


if __name__ == '__main__':
    import time
    import glob
    import os

    l = glob.glob(os.path.join('../sleep_data/sleepedf-39', '*.npz'))
    l = np.array_split(l, 5)

    t1 = time.time()
    load_npz_files(glob.glob(os.path.join('../sleep_data/sleepedf-39', '*.npz')), 1)
    t2 = time.time()
    print(f"cost: {t2-t1}")
