# 2021.7.28 Poisson-RTN

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from ops_laplace_new import *
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

        self.conv0 = nn.Conv1d(in_channels=window_size,out_channels = window_size, kernel_size = 2, stride = 2)
        #Spatial-RTN
        self.Srtn = clones(RTN(h_dim=self.hidden_size,
                       graph_node_dim=self.graph_node_dim,
                       layer_num_rnn=self.layer_num_rnn,
                       window_size=window_size,
                       dropout=0.1,
                       res=res), self.heads)
        #Temporal-RTN
        self.Prtn = clones(RTN(h_dim=self.hidden_size * self.window_size // 2,
                       graph_node_dim=self.graph_node_dim,
                       layer_num_rnn=self.layer_num_rnn,
                       window_size=time_series,
                       dropout=0.1,
                       res=res), self.heads)
        self.layer2 = nn.Sequential(
            nn.Linear(self.time_series * (self.hidden_size * self.window_size // 2) // 2, last_dense),
            #nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(last_dense, output_size))
        #
        self.Snorm = LayerNorm(self.hidden_size // 2)
        self.Snorm2 = LayerNorm(self.hidden_size // 2)
        self.Pnorm = LayerNorm((self.hidden_size * self.window_size // 2) // 2)
        self.Pnorm2 = LayerNorm((self.hidden_size * self.window_size // 2) // 2)
        #
        self.conv1 = nn.Conv1d(in_channels=window_size,out_channels = window_size, kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv1d(in_channels=time_series,out_channels = time_series, kernel_size = 2, stride = 2)
        #
        self.SMultiLx = nn.Linear(self.heads * self.hidden_size // 2, self.hidden_size // 2)
        self.PMultiLx = nn.Linear(self.heads * (self.hidden_size * self.window_size // 2) // 2, (self.hidden_size * self.window_size // 2) // 2)
        #FFN
        self.Sffn = PositionwiseFeedForward(d_model = self.hidden_size // 2, d_ff = 128, dropout=0.1)
        self.Pffn = PositionwiseFeedForward(d_model = (self.hidden_size * self.window_size // 2) // 2, d_ff = 128, dropout=0.1)
        #
        self.Sdropout1 = nn.Dropout(0.1)
        self.Sdropout2 = nn.Dropout(0.1)
        self.Pdropout1 = nn.Dropout(0.1)
        self.Pdropout2 = nn.Dropout(0.1)
        #PE
        #self.Spe = PositionalEncoding(d_model = self.window_size, dropout = 0.1)
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
        #x = self.pe(x)
        x_first = x
        x_first = self.conv1(x_first)
        #print(x.shape)
        #multi-head-spatial-RTN
        x = x.transpose(2, 1)
        #x = self.Spe(x)
        x = x.transpose(2, 1)
        result_dic = self.Srtn[0](x)
        #print(result_dic["kl_b"].shape)
        klb=result_dic["kl_b"]
        kll=result_dic["kl_l"]
        S_klb = result_dic["S_kl_b"]
        S_kll = result_dic["S_kl_l"]
        g1 = result_dic['summ_graph']
        g2 = result_dic['spec_graph']
        #print(g1.shape)
        rd = []
        for i in range(1, self.heads):
            rd.append(self.Srtn[i](x))
        for k,v in result_dic.items():
            for d in rd:
                klb = klb + d["kl_b"]
                kll = kll + d["kl_l"]
                S_klb = S_klb + d["S_kl_b"]
                S_kll = S_kll + d["S_kl_l"]
                g1 = g1 + d['summ_graph']
                g2 = g2 + d["spec_graph"]
                for m,n in d.items():
                    if k == m:
                        if k == 'ht_output':
                            result_dic[k]= torch.cat([result_dic[k], d[k]], -1)
        #
        S_klb = S_klb.reshape(b, p, -1)[:, p // 2, :]
        S_kll = S_kll.reshape(b, p, -1)[:, p // 2, :]
        S_klb = torch.sum(torch.abs(S_klb))
        S_kll = torch.sum(torch.abs(S_kll))
        #print(S_kll.shape)
        g1 = g1.reshape(b, p, -1)[:, p // 2, :]
        g2 = g2.reshape(b, p, -1)[:, p // 2, :]

        result_dic["kl_b"] = klb / self.heads
        result_dic["kl_l"] = kll / self.heads
        #
        result_dic['summ_graph'] = g1 / self.heads
        result_dic['spec_graph'] = g2 / self.heads
        #
        result_dic["S_kl_b"] = S_klb / self.heads
        result_dic["S_kl_l"] = S_kll / self.heads
        result_dic['ht_output'] = self.SMultiLx(result_dic['ht_output'])
        #
        result_dic['ht_output'] = self.Sdropout1(result_dic['ht_output']) + x_first
        result_dic['ht_output'] = self.Snorm(result_dic['ht_output'])

        x_second = result_dic['ht_output']
        #ffn
        result_dic['ht_output'] = self.Sffn(result_dic['ht_output'])

        #
        result_dic['ht_output'] = self.Sdropout2(result_dic['ht_output']) + x_second
        result_dic['ht_output'] = self.Snorm2(result_dic['ht_output'])
        #
        x = result_dic['ht_output']#.permute(1, 0, 2)
        #######################################
        ##################时间RTN##############
        x = x.reshape(b, self.time_series, -1)
        #print("xx:",x.shape)
        #print(self.hidden_size * self.window_size // 2)
        x_first = x
        x_first = self.conv2(x_first)
        #print('x',x.shape)
        x = x.transpose(2, 1)
        x = self.Tpe(x)
        x = x.transpose(2, 1)
        P_dic = self.Prtn[0](x)
        klb=P_dic["kl_b"]
        kll=P_dic["kl_l"]
        S_klb = P_dic["S_kl_b"]
        S_kll = P_dic["S_kl_l"]
        g1 = P_dic['summ_graph']
        g2 = P_dic['spec_graph']
        #print(g1.shape)
        rd = []
        for i in range(1, self.heads):
            rd.append(self.Prtn[i](x))
        for k,v in P_dic.items():
            for d in rd:
                klb = klb + d["kl_b"]
                kll = kll + d["kl_l"]
                S_klb = S_klb + d["S_kl_b"]
                S_kll = S_kll + d["S_kl_l"]
                g1 = g1 + d['summ_graph']
                g2 = g2 + d["spec_graph"]
                for m,n in d.items():
                    if k == m:
                        if k == 'ht_output':
                            P_dic[k]= torch.cat([P_dic[k], d[k]], -1)

        P_dic["kl_b"] = klb / self.heads
        P_dic["kl_l"] = kll / self.heads
        P_dic['summ_graph'] = g1 / self.heads
        P_dic['spec_graph'] = g2 / self.heads
        P_dic["S_kl_b"] = S_klb / self.heads
        P_dic["S_kl_l"] = S_kll / self.heads
        #print(P_dic["ht_output"].shape)
        P_dic['ht_output'] = self.PMultiLx(P_dic['ht_output'])
        #print(P_dic["ht_output"].shape)
        #print(P_dic["summ_graph"].shape)

        #
        P_dic['ht_output'] = self.Pdropout1(P_dic['ht_output']) + x_first
        P_dic['ht_output'] = self.Pnorm(P_dic['ht_output'])

        x_second = P_dic['ht_output']
        #ffn
        P_dic['ht_output'] = self.Pffn(P_dic['ht_output'])

        #
        P_dic['ht_output'] = self.Pdropout2(P_dic['ht_output']) + x_second
        P_dic['ht_output'] = self.Pnorm2(P_dic['ht_output'])
        x = P_dic['ht_output']
        result_dic['P_summ_graph'] = P_dic['summ_graph']
        result_dic['P_spec_graph'] = P_dic['spec_graph']
        result_dic['S_summ_graph'] = result_dic['summ_graph']
        result_dic['S_spec_graph'] = result_dic['spec_graph']
        result_dic["P_kl_b"] = P_dic["kl_b"]
        result_dic["P_kl_l"] = P_dic["kl_l"]
        #loss
        result_dic["kl_b"] = result_dic["P_kl_b"] + result_dic["S_kl_b"]
        result_dic["kl_l"] = result_dic["P_kl_l"] + result_dic["S_kl_l"]
        b, t, h = x.size()
        x = torch.reshape(x, (b, t * h))
        x = self.layer2(x)
        result_dic['ht_output'] = x
        #print(result_dic)
        return result_dic