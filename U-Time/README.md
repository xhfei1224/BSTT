# U-Time

U-Time model implemented by pytorch.

## Datasets
>We evaluate the performance of SalientSleepNet on Sleep-EDF-39 and Sleep-EDF-153 datasets, which are open-access databases of polysomnography (PSG) recordings.

## Requirements
* Python 3.7
* Pytorch 1.11.0
* Numpy 1.21.2
* Matplotlib 3.5.1
* mne 0.20.1
* tqdm 4.62.3
* torchinfo 1.6.3
* pandas 1.3.5

## Function of file
* `prepare_npz`
  * Prepare npz files from raw data.
* `util/load_files.py`
  * Provide functions to load sleep signal files (npz files).
* `util/data.py`
  * Define Sleep Dataset Class.
* `preprocess.py`
  * Provide functions to preprocess sleep signal sequence.
* `model.py`
  * Generate U-Time model.
* `train.py`
  * Training the model.

## Usage
### 1.Get data
You can get Sleep-EDF-39 & Sleep-EDF-153 with:
>$ wget https://www.physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip

### 2.Data preparation
>$ python prepare_npz/prepare_npz_data -d ./sleep-edf-database-expanded/sleep-cassette -o ./data/sleep-edf-153/npzs
* `--data_dir -d` File path to the edf file that contain sleeping info.
* `--output_dir -o` Directory where to save outputs.
* `--select_ch -s` Choose the channels for training.

### 3.Training
>$ python ./train.py -d "./data/sleep-edf-153/npzs" -b 12
* `--data_dir -d` The address of data (directory of npz files).
* `--batch_size -b` Define the training batch size.
* `--fold -f` Define the K-fold validation folds.
* `--window_size -t` Define the training sequence length.