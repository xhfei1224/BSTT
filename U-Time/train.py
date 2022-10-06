from model import UTime
from preprocess import get_eeg_datasets
from argparse import ArgumentParser
from tqdm import tqdm
from functools import reduce
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
from torch import optim
from typing import Tuple
import sklearn.metrics as metrics
def PrintScore(true, pred, savePath=None, average='macro',
               classes=['Wake', 'N1', 'N2', 'N3', 'REM'], learning_info = None):
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3], F1[4]),
          file=saveFile)
    # Classification report
    print()
    print("Classification report:", file=saveFile)
    print(metrics.classification_report(true, pred, target_names=classes, digits=4),
          file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)
    # Overall scores
    print()
    print('    Accuracy\t', metrics.accuracy_score(true, pred), file=saveFile)
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred), file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=average),
          '\tAverage =', average, file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=average),
          '\tAverage =', average, file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=average),
          '\tAverage =', average, file=saveFile)
    # Results of each class
    print('\nResults of each class:', file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=None), file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=None), file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=None), file=saveFile)
    #learning_info
    print('learning_info:\n', learning_info, file = saveFile)
    if savePath != None:
        saveFile.close()
    return
othercfg = {
    'fold': 5,
    'data_path': '../MASS_5CS_5Slice_new/',#'../ISRUC_5CS_5Slice_new/',#'../ISRUC_5CS_5Slice_new/',#'../ISRUC_5CS_feature/', #../MASS_5CS_feature/
    'out_dir': './MASS_result/',
    'workers': 0,
}

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y), \
            'The number of inputs(%d) and targets(%d) does not match.' % (len(x), len(y))
        self.x = x
        self.y = y
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def parse_arguments() -> Tuple[str, int, int, int, int]:
    parser = ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, 
                        default='../../sleep_data/sleepedf-39')
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--train_epoch', '-e', type=int, default=40)
    parser.add_argument('--folds', '-f', type=int, default=5)
    parser.add_argument('--window_size', '-t', type=int, default=19)
    args = parser.parse_args()
    return args.data_dir, args.batch_size, args.train_epoch, args.folds, args.window_size


def concat_dataset(l) -> Dataset:
    return reduce(lambda x, y: x + y, l) if len(l) > 1 else l[0]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # hyper-parameters & data
    data_path, batch_size, epoch, folds, window_size = parse_arguments()
    #dataset_list = get_eeg_datasets(data_path, stride=window_size)
    #assert len(dataset_list) > folds, "too less data for training and validation"
    #for dataset in dataset_list:
    #    dataset.normalization()
    #split_data_list = np.array_split(np.asarray(dataset_list, dtype=object), folds)

    # model
    model = UTime(window_size).to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    # training fold
    total_acc = []
    total_f1 = []
    total_kappa = []
    fold_acc = []
    fold_f1 = []
    fold_kappa = []
    tmpperd = np.zeros(0)
    tmptar = np.zeros(0)
    for k in range(folds):
        print(f"\n=======================================✨ fold: {k+1} / {folds} "
              f"===============================================")
        '''
        train_set = list(set(dataset_list) - set(split_data_list[k]))
        val_set = split_data_list[k]
        train_set, val_set = concat_dataset(train_set), concat_dataset(val_set)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        '''
        trainX = []
        trainY = []
        valX = []
        valY = []
        for fid in range(othercfg['fold']):
            if fid!=4 - k:
                trainX.append(np.float32(np.load(othercfg['data_path'] + 'valX_F' + str(fid) + '.npy')))
                trainY.append(np.load(othercfg['data_path'] + 'valY_F' + str(fid) + '.npy'))
            else:
                valX.append(np.float32(np.load(othercfg['data_path'] + 'valX_F' + str(fid) + '.npy')))
                valY.append(np.load(othercfg['data_path'] + 'valY_F' + str(fid) + '.npy'))

        trainX = np.concatenate(trainX)
        trainY = np.concatenate(trainY)
        valX = np.concatenate(valX)
        valY = np.concatenate(valY)
        print('[Data] Train:', len(trainY), 'Val:', len(valY))
        print('[Data] Train:', trainY.shape, 'Val:', valY.shape)
        trDataset = SimpleDataset(np.float32(trainX), trainY)
        cvDataset = SimpleDataset(np.float32(valX), valY)
        train_loader = DataLoader(
                trDataset, batch_size=batch_size,
                shuffle=True, num_workers=0, drop_last=True
            )
        val_loader = DataLoader(
                cvDataset, batch_size=batch_size,
                shuffle=False, num_workers=0, drop_last=True
            )
        per_epoch_train_mean_loss, per_epoch_val_mean_loss = [], []
        per_epoch_train_acc, per_epoch_val_acc = [], []
        tmp_acc = 0
        tmp_f1 = 0
        tmp_kappa = 0
        # training epoch
        for e in range(epoch):
            print(f"\n-----------------------------------------🐍[{e+1}/{epoch}]-------------------------"
                  f"----------------------")

            # training
            model.train()
            total_num = 0
            correct_num = 0
            loss_list = []
            with tqdm(train_loader, unit='batch') as tepoch:
                for x, target in tepoch:
                    tepoch.set_description("Training")
                    x, target = x.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(x)
                    #print(output.shape)
                    #print(target)
                    loss = loss_func(output, target.long())
                    loss.backward()
                    optimizer.step()

                    predict = output.argmax(1)

                    correct_num += predict.eq(target.view_as(predict)).sum().item()
                    total_num += len(target.flatten())
                    loss_list.append(loss.item())
                    tepoch.set_postfix(loss=np.mean(loss_list), accuracy=f'{correct_num / total_num:.5}')
            per_epoch_train_mean_loss.append(np.mean(loss_list))
            per_epoch_train_acc.append(correct_num / total_num)

            # validation
            model.eval()
            total_validation_loss = 0
            total_validation_correct = 0
            total_num = 0
            pred = 0
            
            tar = 0
            with torch.no_grad():
                for x, target in val_loader:
                    x, target = x.to(device), target.to(device)
                    output = model(x)
                    loss = loss_func(output, target.long())
                    total_validation_loss += loss.item()
                    total_validation_correct += output.argmax(1).eq(target).sum().item()
                    try:
                        if pred == 0:
                            pred = output.argmax(1)
                            #tmpperd = pred
                        if tar == 0:
                            tar = target
                            #tmptar = tar
                    except:
                        pred = torch.cat((pred, output.argmax(1)), dim = -1)
                        tar = torch.cat((tar,target), dim = -1)
                    total_num += len(target.flatten())
            pred = pred.cpu().numpy()
            tmpperd = np.concatenate((tmpperd, pred))
            tar = tar.cpu().numpy()
            tmptar = np.concatenate((tmptar, tar))
            mean_loss = total_validation_loss / len(val_loader)
            acc = total_validation_correct / total_num
            if acc > tmp_acc:
                tmp_acc = acc
                tmp_f1 = metrics.f1_score(tar, pred, average = 'macro')
                tmp_kappa = metrics.cohen_kappa_score(tar, pred)
            print(f"\n[Validation-{e+1}] loss {mean_loss:.8} accuracy: {acc:.8}\n")
            per_epoch_val_mean_loss.append(mean_loss)
            per_epoch_val_acc.append(acc)
        fold_acc.append(tmp_acc)
        fold_f1.append(tmp_f1)
        fold_kappa.append(tmp_kappa)
        print(fold_acc)
        print(fold_f1)
        print(fold_kappa)
        PrintScore(tmpperd, tmptar, savePath = 'result/')
        train_loss.append(per_epoch_train_mean_loss)
        train_acc.append(per_epoch_train_acc)
        val_loss.append(per_epoch_val_mean_loss)
        val_acc.append(per_epoch_val_acc)
        torch.save(model, f"{model.__class__.__name__}-fold-{k}.pth")
        model.reset_parameters()
    total_acc = sum(fold_acc) / 5.0
    total_f1 = sum(fold_f1) / 5.0
    total_kappa = sum(fold_kappa) / 5.0
    save_dict = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    filename = 'output.txt'
    with open(filename,'a') as file_object:
        file_object.write('\nacc:')
        file_object.write(str(total_acc))
        file_object.write('f1:')
        file_object.write(str(total_f1))
        file_object.write('kappa:')
        file_object.write(str(total_kappa))
    np.savez(f"{model.__class__.__name__}_training_data.npz", **save_dict)
