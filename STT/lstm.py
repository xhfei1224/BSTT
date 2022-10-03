# 2021.7.28 Poisson-RTN

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

#from ops_laplace_new import *
from feature import *




class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num_rnn,
                 graph_node_dim, window_size, last_dense=128, res=True, heads = 5, time_series = 5):
        super(lstm, self).__init__()
        self.first_hidden_size = hidden_size
        self.hidden_size = hidden_size // 2
        self.layer_num_rnn = layer_num_rnn
        self.graph_node_dim = graph_node_dim
        self.window_size = window_size
        self.res = res
        self.heads = heads
        self.layer1 = TinyFeature(s_freq=100, filters=128, dropout=0.5)
        self.time_series = time_series

        if self.first_hidden_size!=512:
            self.layer1t = nn.Linear(512, self.first_hidden_size)
        #
        self.conv0 = nn.Conv1d(in_channels=window_size,out_channels = window_size, kernel_size = 2, stride = 2)
        #Mutil-head-attention
        self.Srtn = MultiHeadedAttention(h = heads, d_model = self.hidden_size, dropout=0.1)
        #Mutil-head-attention
        self.Prtn = MultiHeadedAttention(h = heads, d_model = self.hidden_size * self.window_size, dropout=0.1) 
        self.layer2 = nn.Sequential(
            nn.Linear(self.time_series * (self.hidden_size * self.window_size // 2) // 2, last_dense),
            #nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(last_dense, output_size))
        self.PHL = nn.Linear(self.hidden_size * self.window_size // 2, self.hidden_size * self.window_size // 4)
        #
        self.Snorm = LayerNorm(self.hidden_size // 2)
        self.Snorm2 = LayerNorm(self.hidden_size)
        self.Pnorm = LayerNorm((self.hidden_size * self.window_size // 2)  // 2)
        self.Pnorm2 = LayerNorm((self.hidden_size * self.window_size // 2) // 2)
        #
        self.conv1 = nn.Conv1d(in_channels=window_size,out_channels = window_size, kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv1d(in_channels=time_series,out_channels = time_series, kernel_size = 2, stride = 2)
        #
        self.SMultiLx = nn.Linear(self.heads * self.hidden_size // 2, self.hidden_size // 2)
        self.PMultiLx = nn.Linear(self.heads * (self.hidden_size * self.window_size // 2) // 2, (self.hidden_size * self.window_size // 2) // 2)
        #FFN
        self.Sffn = PositionwiseFeedForward(d_model = self.hidden_size, d_ff = 128, dropout=0.1)
        self.Pffn = PositionwiseFeedForward(d_model = (self.hidden_size * self.window_size // 2) // 2, d_ff = 128, dropout=0.1)
        #
        self.Sdropout1 = nn.Dropout(0.1)
        self.Sdropout2 = nn.Dropout(0.1)
        self.Pdropout1 = nn.Sequential(
                        nn.Linear(self.hidden_size * self.window_size, self.hidden_size * self.window_size // 2),
                        nn.Dropout(0.1))
                        
        self.Pdropout2 = nn.Dropout(0.1)
        #PE
        self.Spe = PositionalEncoding(d_model = self.window_size, dropout = 0.1)
        self.Tpe = PositionalEncoding(d_model = self.time_series, dropout = 0.1)
    def forward(self, x):

        b, d, t, h = x.size()
        p = d
        #print("{},{},{},{}".format(b, d, t, h))
        x = torch.reshape(x, (b*d*t, 1, h))
        x = self.layer1(x)
        if self.first_hidden_size!=512:
            x = self.layer1t(x)
        
        x = torch.reshape(x, (b * d, t, self.first_hidden_size))
        x = self.conv0(x)
        #
        x_first = x
        x_first = self.conv1(x_first)
        
        #print(x.shape)
        #mutil-head attention
        x = x.transpose(2, 1)
        x = self.Spe(x)
        x = x.transpose(2, 1)

        x = self.Srtn(x, x, x)
        x_second = x
        #ffn
        #print(x.shape)
        x = self.Sffn(x)

        #
        x = self.Sdropout2(x) + x_second
        x = self.Snorm2(x)
        #
        #x = result_dic['ht_output']#.permute(1, 0, 2)
        ############################################
        ##################Temporal-RTN##############
        x = x.reshape(b, self.time_series, -1)
        #print("xx:",x.shape)
        #print(self.hidden_size * self.window_size // 2)
        x_first = x
        x_first = self.conv2(x_first)
        #print('x',x.shape)
        x = x.transpose(2, 1)
        x = self.Tpe(x)
        x = x.transpose(2, 1)
        #print(x.shape)
        x = self.Prtn(x,x,x)
        
        #1
        x = self.Pdropout1(x) + x_first
        x = self.PHL(x)
        x = self.Pnorm(x)

        x_second = x
        #ffn
        #print(x.shape)
        x = self.Pffn(x)
        #print(P_dic["ht_output"].shape)

        #2
        x = self.Pdropout2(x) + x_second
        x = self.Pnorm2(x)

        b, t, h = x.size()
        x = torch.reshape(x, (b, t * h))
        x = self.layer2(x)
        #print("xxx:",x.shape)

        #print(result_dic)
        return x