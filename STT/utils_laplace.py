# 2021.7.28 Poisson-RTN

import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

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


def train_epoch(model, train_loader, learning, my_loss, optimizer, epoch, threshold=10):
    model.train()
    acc = 0
    total = 0
    loss_list = []
    loss_kl1_list = []
    loss_kl2_list = []
    loss_kl3_list = []
    loss_kl4_list = []

    for batch_idx, (x, y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()
        b, _, _, _ = x.size()

        optimizer.zero_grad()
        result_dic = {"ht_output": 0}
        result_dic["ht_output"] = model(x)
        #torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)

        loss_kl1 = None#learning['lamada1'] * result_dic['S_kl_l']
        loss_kl2 = None#learning['lamada2'] * result_dic['S_kl_b']
        loss_kl3 = None#learning['lamada3'] * result_dic['P_kl_l']
        loss_kl4 = None#learning['lamada4'] * result_dic['P_kl_b']
        loss = my_loss(result_dic["ht_output"], y)
        #print(loss)
        loss.backward()
        #with torch.autograd.detect_anomaly():
        #    loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()

        pred = torch.argmax(result_dic['ht_output'].data, 1)
        if learning['smooth']:
            y = torch.argmax(y, 1)
        acc += ((pred == y).sum()).cpu().numpy()
        total += len(y)
        loss_list.append(loss.item())
        #print("[TR]epoch:%d, step:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
        #      (epoch + 1, batch_idx, loss, result_dic['kl_g'], result_dic['kl_b'], (acc / total)))
    loss_kl1_mean = 0
    loss_kl2_mean = 0
    loss_kl3_mean = 0
    loss_kl4_mean = 0
    loss_mean = np.mean(loss_list)
    print("[TR]epoch:%d, loss:%f, l1:%f, l2:%f, l3:%f, l4:%f, acc:%f" %
          (epoch + 1, loss_mean, loss_kl1_mean, loss_kl2_mean, loss_kl3_mean, loss_kl4_mean, (acc / total)))
    return acc / total, loss_mean, loss_kl1_mean, loss_kl2_mean, loss_kl3_mean, loss_kl4_mean


def val(model, val_loader, learning, my_loss, optimizer, epoch):
    model.eval()
    acc = 0
    total = 0
    pred_list = []
    true_list = []
    loss_list = []
    loss_kl1_list = []
    loss_kl2_list = []
    loss_kl3_list = []
    loss_kl4_list = []
    S_summ_graph_list = []
    S_spec_graph_list = []
    P_summ_graph_list = []
    P_spec_graph_list = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            # If you run this code on CPU, please remove the '.cuda()'
            x = x.cuda()
            y = y.cuda()
            b, _, _, _ = x.size()

            optimizer.zero_grad()
            result_dic = {"ht_output": 0}
            result_dic["ht_output"] = model(x)

            loss_kl1 = None#learning['lamada1'] * result_dic['S_kl_l']
            loss_kl2 = None#learning['lamada2'] * result_dic['S_kl_b']
            loss_kl3 = None#learning['lamada3'] * result_dic['P_kl_l']
            loss_kl4 = None#learning['lamada4'] * result_dic['P_kl_b']
            loss = my_loss(result_dic['ht_output'], y)

            pred = torch.argmax(result_dic['ht_output'].data, 1)
            if learning['smooth']:
                y = torch.argmax(y, 1)
            pred_list.append(pred.cpu().numpy())
            true_list.append(y.cpu().numpy())
            acc += ((pred == y).sum()).cpu().numpy()
            total += len(y)

            loss_list.append(loss.item())


            S_summ_graph_list.append(None)
            S_spec_graph_list.append(None)
            P_summ_graph_list.append(None)
            P_spec_graph_list.append(None)
            #print("val:    epoch:%d ,step:%d, total loss:%f, kl loss1 is %f, kl loss2 is %f"
            #      %(epoch+1,batch_idx,loss,result_dic['kl_g'],result_dic['kl_b']), end=', ')
    pred_list = np.concatenate(pred_list)
    true_list = np.concatenate(true_list)
    loss_mean = np.mean(loss_list)
    loss_kl1_mean = 0
    loss_kl2_mean = 0
    loss_kl3_mean = 0
    loss_kl4_mean = 0
    S_summ_graph_list = None
    S_spec_graph_list = None
    P_summ_graph_list = None
    P_spec_graph_list = None

    print("[VA]epoch:%d, loss:%f, l1:%f, l2:%f, l3:%f, l4:%f, acc:%f" %
          (epoch + 1, loss_mean, loss_kl1_mean, loss_kl2_mean, loss_kl3_mean, loss_kl4_mean,(acc / total)), end='')
    return acc / total, loss_mean, loss_kl1_mean, loss_kl2_mean, loss_kl3_mean, loss_kl4_mean,\
           pred_list, true_list, S_summ_graph_list, S_spec_graph_list,\
           P_summ_graph_list, P_spec_graph_list


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


def ConfusionMatrix(y_true, y_pred, savePath=None, title=None, cmap=plt.cm.Blues,
                    classes=['Wake', 'N1', 'N2', 'N3', 'REM']):
    if not title:
        title = 'Confusion matrix'
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n = cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Confusion matrix")
    print(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i,
                    format(cm[i, j] * 100, '.2f') + '%\n' +
                    format(cm_n[i, j], 'd'),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if savePath is not None:
        plt.savefig(savePath + title + ".png")
    plt.show()
    return ax

def row2matrix(x, side):
    result = np.empty([side,side])
    edge_idx = 0
    for i in range(side):
        for j in range(i):
            result[i,j] = x[edge_idx]
            result[j,i] = x[edge_idx]
            edge_idx += 1
        result[i,i] = np.nan
    return result