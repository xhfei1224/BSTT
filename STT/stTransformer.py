# 2021.7.28 Poisson-RTN
import os

from torch._C import device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import time
import random
import gc
from torch.utils.data import DataLoader
#from torchsummary import summary
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
from ops_laplace_new import *
from lstm import *
from utils_laplace import *
torch.set_num_threads(6)

#from lib.ops import Dense,RTN
#from dataGenSequences_rtn import dataGenSequences
#from compute_priors import compute_priors

print('=====Start=====', time.asctime(time.localtime(time.time())))
print('PyTorch:', torch.__version__)

####################### learning config ########################
learning = {
    'optimizer': 'AdamW',  #'SGD'
    'lr': 0.0008, #0.003151,  #0.002,
    'lr_decay': 1.2,
    'weight_decay': 0.00003,
    'batchSize': 256,
    'minEpoch': 60,
    'feaDim': 0,
    'hiddenDim': 256, #256
    'layer_num_rnn': 5,
    'window_size': 6,
    'graph_dim': 256, #256
    'lamada1': 0.00005,
    'lamada2': 0.00004,
    'lamada3': 0.00005,
    'lamada4': 0.00004,
    'targetDim': 5,
    'seed': 1,
    'heads':3,
    'time_series':5,
    'smooth': True,
    'oversamN1': False,
    'oversamREM': False,
    'res': False
}
othercfg = {
    'fold': 5,
    'data_path': '../ISRUC_5CS_5Slice_new/',#'../ISRUC_5CS_feature/', #../MASS_5CS_feature/
    'out_dir': './ISRUC_result/',
    'workers': 0,
}
print('[Info] Config:')
print('learning:', learning)
print('othercfg:', othercfg)
if not os.path.exists(othercfg['out_dir']):
    os.makedirs(othercfg['out_dir'])
    print('[Info] Make out dir:', othercfg['out_dir'])

####################### Set random seed ########################
seed = learning['seed']
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.cuda.set_device(0)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpu
torch.backends.cudnn.deterministic = True

########################### Training ###########################
pred_list = []
true_list = []
S_summ_graph_list=[]
S_spec_graph_list=[]
P_summ_graph_list=[]
P_spec_graph_list=[]
tr_acc_list = []
tr_loss_list = []
tr_kl1_list = []
tr_kl2_list = []
tr_kl3_list = []
tr_kl4_list = []
val_acc_list = []
val_loss_list = []
val_kl1_list = []
val_kl2_list = []
val_kl3_list = []
val_kl4_list = []
wake_spec = []
rem_spec = []
n1_spec = []
n2_spec = []
n3_spec = []
wake_summ = []
rem_summ = []
n1_summ = []
n2_summ = []
n3_summ = []
twake_spec = []
trem_spec = []
tn1_spec = []
tn2_spec = []
tn3_spec = []
twake_summ = []
trem_summ = []
tn1_summ = []
tn2_summ = []
tn3_summ = []
tn1_trans_spec = []
tn1_trans_summ = []
tn2_trans_spec = []
tn2_trans_summ = []
tn3_trans_spec = []
tn3_trans_summ = []
trem_trans_spec = []
trem_trans_summ = []
twake_trans_spec = []
twake_trans_summ = []

wake2n1_spec = []
wake2n1_summ = []
wake2n2_spec = []
wake2n2_summ = []
wake2n3_spec = []
wake2n3_summ = []
wake2rem_spec = []
wake2rem_summ = []

n12n2_spec = []
n12n2_summ = []
n12n3_spec = []
n12n3_summ = []
n12rem_spec = []
n12rem_summ = []
n12wake_spec = []
n12wake_summ = []

n22n1_spec = []
n22n1_summ = []
n22n3_spec = []
n22n3_summ = []
n22rem_spec = []
n22rem_summ = []
n22wake_spec = []
n22wake_summ = []

n32n2_spec = []
n32n2_summ = []
n32n1_spec = []
n32n1_summ = []
n32rem_spec = []
n32rem_summ = []
n32wake_spec = []
n32wake_summ = []

rem2wake_spec = []
rem2wake_summ = []
rem2n1_spec = []
rem2n1_summ = []
rem2n2_spec = []
rem2n2_summ = []
rem2n3_spec = []
rem2n3_summ = []


print('[Info]', time.asctime(time.localtime(time.time())), 'Start training:')
for i in range(othercfg['fold']):
    print()
    print(24 * "=", 'Fold #', i, 'Train', 24 * '=')
    print(time.asctime(time.localtime(time.time())))
    # Load data from .npy files
    trainX = []
    trainY = []
    valX = []
    valY = []
    for fid in range(othercfg['fold']):
        if fid!=4 - i:
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
    # Organize data to Torch
    if learning['oversamN1']:
        trainX = np.concatenate([trainX, trainX[trainY==1]])
        trainY = np.concatenate([trainY, trainY[trainY==1]])
    if learning['oversamREM']:
        trainX = np.concatenate([trainX, trainX[trainY==4]])
        trainY = np.concatenate([trainY, trainY[trainY==4]])
    if learning['smooth']:
        trainY = np.eye(5)[trainY]
        valY = np.eye(5)[valY]
        trainY = torch.tensor(trainY)
        valY = torch.tensor(valY)
        trainY = smooth_one_hot(trainY, classes=5, smoothing=0.1)
        valY = smooth_one_hot(valY, classes=5, smoothing=0.1)
        print('[Smooth] Train:', trainY.shape, 'Val:', valY.shape)
    trDataset = SimpleDataset(np.float32(trainX), trainY)
    cvDataset = SimpleDataset(np.float32(valX), valY)
    trGen = DataLoader(trDataset,
                       batch_size=learning['batchSize'],
                       shuffle=True,
                       num_workers=othercfg['workers'])
    cvGen = DataLoader(cvDataset,
                       batch_size=learning['batchSize'],
                       shuffle=False,
                       num_workers=othercfg['workers'])

    # Define Model\Loss\Optimizer
    model = lstm(input_size=learning['feaDim'],
                 hidden_size=learning['hiddenDim'],
                 output_size=learning['targetDim'],
                 layer_num_rnn=learning['layer_num_rnn'],
                 graph_node_dim=learning['graph_dim'],
                 window_size=learning['window_size'],
                 res=learning['res'],
                 heads = learning['heads']).cuda()
    #get_name_par(model)
    print(get_parameter_number(model))
    
    if learning['smooth']:
        loss_func = one_hot_CrossEntropy()
    else:
        loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1.5,1,1,1]).cuda())
    
    if learning['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning['lr'], weight_decay=learning['weight_decay'])  #,momentum=0.5,nesterov=True)
    elif learning['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning['lr'], weight_decay=learning['weight_decay'])
    elif learning['optimizer'] == 'Adam+AMS':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning['lr'], weight_decay=learning['weight_decay'], amsgrad=True)
    elif learning['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning['lr'], weight_decay=learning['weight_decay'], amsgrad=False)
    elif learning['optimizer'] == 'AdamW+AMS':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning['lr'], weight_decay=learning['weight_decay'], amsgrad=True)
    elif learning['optimizer'] == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning['lr'], weight_decay=learning['weight_decay'])
    elif learning['optimizer'] == 'NAdam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=learning['lr'], weight_decay=learning['weight_decay'])
    elif learning['optimizer'] == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=learning['lr'], weight_decay=learning['weight_decay'])
    elif learning['optimizer'] == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning['lr'], weight_decay=learning['weight_decay'])
    if learning['lr_decay']>0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=learning['lr_decay'])

    # Train & Valid 1 epoch
    val_loss_before = np.inf
    best_acc = 0
    count_epoch = 0
    tr_acc_list_e = []
    tr_loss_list_e = []
    tr_kl1_list_e = []
    tr_kl2_list_e = []
    tr_kl3_list_e = []
    tr_kl4_list_e = []
    val_acc_list_e = []
    val_loss_list_e = []
    val_kl1_list_e = []
    val_kl2_list_e = []
    val_kl3_list_e = []
    val_kl4_list_e = []
    for epoch in range(learning['minEpoch']):
        time_start = time.time()

        tr_acc, tr_loss, tr_kl1, tr_kl2, tr_kl3, tr_kl4 = train_epoch(model, trGen, learning, loss_func, optimizer, epoch)
        va_acc, va_loss, va_kl1, va_kl2, va_kl3, va_kl4, _, _, _, _, _, _ = val(model, cvGen, learning, loss_func, optimizer, epoch)

        if learning['lr_decay']>0:
            lr_epoch = scheduler.get_last_lr()
        else:
            lr_epoch = learning['lr']#scheduler.get_last_lr()
        tr_acc_list_e.append(tr_acc)
        tr_loss_list_e.append(tr_loss)
        tr_kl1_list_e.append(tr_kl1)
        tr_kl2_list_e.append(tr_kl2)
        tr_kl3_list_e.append(tr_kl3)
        tr_kl4_list_e.append(tr_kl4)
        val_acc_list_e.append(va_acc)
        val_loss_list_e.append(va_loss)
        val_kl1_list_e.append(va_kl1)
        val_kl2_list_e.append(va_kl2)
        val_kl3_list_e.append(va_kl3)
        val_kl4_list_e.append(va_kl4)

        if val_loss_before < va_loss:
            if count_epoch >= 2:
                if learning['lr_decay']>0:
                    scheduler.step()
                val_loss_before = np.inf
                count_epoch = 0
            else:
                val_loss_before = va_loss
                count_epoch += 1
        else:
            val_loss_before = va_loss

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), othercfg['out_dir'] + 'best_model_' + str(i) + '.nnet.pth')
            print(" U ", end='')
        time_end = time.time()
        time_cost = time_end - time_start
        print(" Time:%.3f" % (time_cost), 'lr:', lr_epoch)

    print(24 * "=", 'Fold #', i, 'Final', 24 * '=')
    print(time.asctime(time.localtime(time.time())))
    model.eval()
    model.load_state_dict(torch.load(othercfg['out_dir'] + 'best_model_' + str(i) + '.nnet.pth'))
    _, _, _, _, _, _, pred, true, S_summ, S_spec, P_summ, P_spec = val(model, cvGen, learning, loss_func, optimizer, epoch)
    pred_list.append(pred)
    true_list.append(true)
    S_summ_graph_list.append(S_summ)
    S_spec_graph_list.append(S_spec)
    P_summ_graph_list.append(P_summ)
    P_spec_graph_list.append(P_spec)
    tr_acc_list.append(tr_acc_list_e)
    tr_loss_list.append(tr_loss_list_e)
    tr_kl1_list.append(tr_kl1_list_e)
    tr_kl2_list.append(tr_kl2_list_e)
    tr_kl3_list.append(tr_kl3_list_e)
    tr_kl4_list.append(tr_kl4_list_e)
    val_acc_list.append(val_acc_list_e)
    val_loss_list.append(val_loss_list_e)
    val_kl1_list.append(val_kl1_list_e)
    val_kl2_list.append(val_kl2_list_e)
    val_kl3_list.append(val_kl3_list_e)
    val_kl4_list.append(val_kl4_list_e)
    
    torch.save(model, othercfg['out_dir'] + 'model_' + str(i) + '.nnet.pth')
    torch.cuda.empty_cache()
    del trainX, valX, trainY, valY, trGen, cvGen, trDataset, cvDataset, loss_func, optimizer
    gc.collect()

true_list = np.concatenate(true_list)
pred_list = np.concatenate(pred_list)


######################## Model metrics #########################
print('\n')
print(10 * '=', 'End Of Training', 10 * '=')
print(time.asctime(time.localtime(time.time())))
PrintScore(true_list, pred_list)
PrintScore(true_list, pred_list, savePath=othercfg['out_dir'], learning_info = learning)
ConfusionMatrix(true_list, pred_list, savePath=othercfg['out_dir'])

##################### Accuracy\Loss curve ######################
plt.figure(figsize=(15, 8))
plt.subplot(2, 2, 1)
for i in range(5):
    plt.plot(tr_acc_list[i], label='Fold %d' % (i))
plt.title('Accuracy')
plt.legend()
plt.subplot(2, 2, 2)
for i in range(5):
    plt.plot(tr_loss_list[i], label='Fold %d' % (i))
plt.title('Total Loss')
plt.legend()
plt.subplot(2, 2, 3)
for i in range(5):
    plt.plot(tr_kl1_list[i], label='Fold %d' % (i))
plt.title('KL1 Loss')
plt.legend()
plt.subplot(2, 2, 4)
for i in range(5):
    plt.plot(tr_kl2_list[i], label='Fold %d' % (i))
plt.title('KL2 Loss')
plt.legend()
plt.suptitle('Training Curve')
plt.tight_layout()
plt.savefig(othercfg['out_dir'] + 'Curve_TR.png')
plt.show()

plt.figure(figsize=(15, 8))
plt.subplot(3, 2, 1)
for i in range(5):
    plt.plot(val_acc_list[i], label='Fold %d' % (i))
plt.title('Accuracy')
plt.legend()
plt.subplot(3, 2, 2)
for i in range(5):
    plt.plot(val_loss_list[i], label='Fold %d' % (i))
plt.title('Total Loss')
plt.legend()
plt.subplot(3, 2, 3)
for i in range(5):
    plt.plot(val_kl1_list[i], label='Fold %d' % (i))
plt.title('KL1 Loss')
plt.legend()
plt.subplot(3, 2, 4)
for i in range(5):
    plt.plot(val_kl2_list[i], label='Fold %d' % (i))
plt.title('KL2 Loss')
plt.legend()
plt.subplot(3, 2, 5)
for i in range(5):
    plt.plot(val_kl3_list[i], label='Fold %d' % (i))
plt.title('KL3 Loss')
plt.legend()
plt.subplot(3, 2, 6)
for i in range(5):
    plt.plot(val_kl4_list[i], label='Fold %d' % (i))
plt.title('KL4 Loss')
plt.legend()
plt.suptitle('Validation Curve')
plt.tight_layout()
plt.savefig(othercfg['out_dir'] + 'Curve_Val.png')
plt.show()
print()


print('[Info] Config:')
print('learning:', learning)
print('===== Succeed =====', time.asctime(time.localtime(time.time())))

