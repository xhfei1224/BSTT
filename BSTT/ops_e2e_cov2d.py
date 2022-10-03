# 2021.7.28 Poisson-RTN

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import init,Conv1d,MaxPool1d,Conv2d, MaxPool2d
from sru import SRU
import math

import warnings

warnings.filterwarnings('ignore')
#


##############################################################################
#################implemenations of the basic functionality####################
##############################################################################


def error(y, pred):
    return torch.mean(torch.ne(y, pred))


def accuracy(y, pred):
    return torch.mean(torch.eq(y, pred))


def clip(x, min, max):
    return torch.clamp(x, min, max)


def floor(x):
    return torch.floor(x).int()


def ceil(x):
    return torch.ceil(x).int()


def sigmoid(x):
    return torch.sigmoid(x)


def relu(x):
    return F.relu(x)


def leaky_relu(x, negative_slope):
    return F.leaky_relu(x, negative_slope=negative_slope)


def softplus(x):
    return F.softplus(x)


def softmax(x):
    return F.softmax(x)


def tanh(x):
    return torch.tanh(x)


def l2_norm(x, epsilon=0.00001):
    square_sum = torch.sum(torch.pow(x, exponent=2))
    norm = torch.sqrt(torch.add(square_sum, epsilon))
    return norm


def l2_norm_2d(x, epsilon=0.00001):
    square_sum = torch.sum(torch.pow(x, exponent=2))
    norm = torch.mean(torch.sqrt(torch.add(square_sum, epsilon)))
    return norm


# we assume afa=beta
def neg_likelihood_gamma(x, afa, epsilon=0.00001):
    norm = torch.add(x, epsilon)
    neg_likelihood = -(afa - 1) * torch.log(norm) + afa * norm
    return torch.mean(neg_likelihood)


# KL(lambda_t||lambda=1)
def kl_exponential(x, epsilon=0.00001):
    norm = torch.add(x, epsilon)
    kl = -torch.log(norm) + norm
    return torch.mean(kl)


def likelihood(x, y, epsilon=0.00001):
    norm = torch.add(x, epsilon)
    kl = -torch.log(norm) + norm * y
    return 0.25 * torch.mean(kl)


def shape(x):
    return x.shape


def reshape(x, shape):
    y = torch.reshape(x, shape).float()
    return y


def Linear_Function(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret


##############################################################################
################implemenations of the Neuro Networks##########################
######################神经网络的实现###########################################
##############################################################################

#tiny实现
class Tiny(Module):
    __constants__ = ['bias', 'features', 'features']

    def __init__(self, s_time, s_freq, in_channels, tiny_drop):
        super(Tiny, self).__init__()
        self.s_time = s_time
        self.s_freq = s_freq
        self.ks = s_freq//2
        self.st = s_freq//8
        self.pad = ((s_freq - 1)*self.st + self.ks - s_freq)//2
        self.in_channels = in_channels
        self.cnn0=nn.Sequential(
            Conv1d(
                in_channels = 6,
                out_channels = 1,
                kernel_size=1,
                stride=1,
                #padding = self.pad
            )
        )
        self.cnn1=nn.Sequential(
            Conv1d(
                in_channels = 1 * in_channels,
                out_channels = 512 * in_channels,
                kernel_size=self.ks,
                stride=self.st,
                #padding = self.pad
            ),
            torch.nn.BatchNorm1d(num_features = 512 * in_channels),
            nn.ReLU(), 
            MaxPool1d(
                kernel_size = 7,
                stride=7,
            ),
            nn.ReLU(),
            nn.Dropout(p = tiny_drop),
        )
        self.cnn2=nn.Sequential(
            Conv1d(
                in_channels = 1,
                out_channels = 1,
                kernel_size=[1,8],
                stride=1,
                #padding = 2
            ),
            torch.nn.BatchNorm2d(num_features = 1),
            nn.ReLU(),
        )
        self.cnn3=nn.Sequential(
            Conv2d(
                in_channels = 1,
                out_channels = 1,
                kernel_size=[1, 8],
                stride=1,
                #padding = 2
            ),
            torch.nn.BatchNorm2d(num_features = 1),
            nn.ReLU(),
        )
        self.cnn4=nn.Sequential(
            Conv2d(
                in_channels = 1,
                out_channels = 1,
                kernel_size=[1, 8],
                stride=1,
            ),
            torch.nn.BatchNorm2d(num_features = 1),
            nn.ReLU(),
            MaxPool2d(
                kernel_size = [1, 4],
                stride=4,
            ),
            nn.ReLU(),
            nn.Dropout(p = tiny_drop),
        )        
        self.cnn5=nn.Sequential(
            Conv1d(
                in_channels = 1,
                out_channels = 6,
                kernel_size=1,
                stride=1,
                #padding = self.pad
            ),
            nn.ReLU()
        )
    def forward(self, x):
        x_res = x.clone()
        x_res = x_res.permute(1, 0, 2)
        #x_res = torch.flatten(x_res, start_dim = 1)
        #x_res = x_res.reshape(x_res.shape[0], 1, -1)
        #print(x_res)
        #print(x_res.shape)
        #x_res = self.cnn0(x_res)
        #print(x_res)
        #print(x_res.shape)
        x_res = self.cnn1(x_res)
        #print(x_res)
        #print(x_res.shape)
        x_res = x_res.reshape(x_res.shape[0], 1, x_res.shape[1], -1)
        x_res = self.cnn2(x_res)
        #print(x_res)
        #print(x_res.shape)
        x_res = self.cnn3(x_res)
        #print(x_res)
        #print(x_res.shape)
        x_res = self.cnn4(x_res)
        #print(x_res)
        #print(x_res.shape)
        x_res = torch.flatten(x_res, start_dim = 1)
        #print(x_res)
        #print(x_res.shape)
        x_res = x_res.reshape(x_res.shape[0], self.in_channels, -1)
        #x_res = x_res.reshape(x_res.shape[0], self.in_channels, -1)
        #print(x_res)
        #print(x_res.shape)
        #x_res = self.cnn5(x_res)
        #print(x_res)
        #print(x_res.shape)
        #x_res = x_res.permute(1, 0, 2)
        #print(x_res.shape)
        return x_res




#全连接层的实现
class Dense(Module):
    __constants__ = ['bias', 'features', 'features']

    def __init__(self, in_features, out_features, bias=True):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    #全连接层的参数初始化
    def reset_parameters(self):
        if (self.in_features == self.out_features):
            init.orthogonal_(self.weight)
        else:
            init.uniform_(self.weight,
                          a=-math.sqrt(1.0 / self.in_features) * math.sqrt(3),
                          b=math.sqrt(1.0 / self.in_features) * math.sqrt(3))
        if self.bias is not None:
            init.uniform_(self.bias, -0, 0)

    def forward(self, input):
        return Linear_Function(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

#神经网络解码
class encode_mean_std(nn.Module):
    def __init__(self, graph_node_dim, h_dim, window_size, total_window_size, dropout=0.1):
        super(encode_mean_std, self).__init__()

        self.graph_node_dim = graph_node_dim
        self.h_dim = h_dim
        self.window_size = window_size
        self.total_window_size = total_window_size
        self.dropout = dropout

        self.enc = nn.Sequential(
            nn.Linear(graph_node_dim, graph_node_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.enc_g = nn.Sequential(
            nn.Linear(graph_node_dim, graph_node_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.enc_b = nn.Sequential(
            nn.Linear(graph_node_dim, self.graph_node_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.enc_mean = nn.Linear(graph_node_dim, 1)
        self.enc_std = nn.Sequential(
            nn.Linear(graph_node_dim, 1),
            #激活函数
            nn.Softplus())

    def forward(self, x):
        enc_ = self.enc(x)
        enc_g = self.enc_g(enc_)
        enc_b = self.enc_b(enc_)
        mean = self.enc_mean(enc_g)
        std = self.enc_std(enc_g)
        return mean, std, enc_b


class RTN(nn.Module):
    def __init__(self, h_dim, graph_node_dim, layer_num_rnn, window_size, dropout, res=False):
        super(RTN, self).__init__()
        self.h_dim = h_dim
        self.graph_node_dim = graph_node_dim
        self.layer_num_res = layer_num_rnn
        self.dropout_rate = dropout
        self.res = res
        self.window_size = window_size
        self.total_window_size = int((window_size * (window_size - 1)) / 2)  # no self-connection

        # recurrence
        #SRU
        if self.res:
            self.sru_res = SRU(input_size=self.graph_node_dim + self.h_dim,
                               hidden_size=self.h_dim,
                               num_layers=self.layer_num_res,
                               dropout=0.1)
        else:
            self.sru_res = SRU(input_size=self.graph_node_dim,
                               hidden_size=self.h_dim,
                               num_layers=self.layer_num_res,
                               dropout=0.1)

        # prior
        self.prior_enc = encode_mean_std(self.graph_node_dim, self.h_dim,
                                        self.window_size,
                                        self.total_window_size,
                                        self.dropout_rate)
        self.prior_mij = nn.Linear(self.graph_node_dim, self.graph_node_dim)

        # post
        self.post_enc = encode_mean_std(self.graph_node_dim, self.h_dim,
                                        self.window_size,
                                        self.total_window_size,
                                        self.dropout_rate)
        self.post_mean_approx_g = nn.Linear(self.graph_node_dim, 1)
        self.post_std_approx_g = nn.Sequential(
            nn.Linear(self.graph_node_dim, 1),
            nn.Softplus())

        #graph
        self.node_emb = nn.Sequential(
            #输入hidden dim，输出图hidden dim
            Dense(h_dim, self.graph_node_dim),
            nn.ReLU())
        self.node_hidden = nn.Sequential(
            Dense(h_dim, self.graph_node_dim),
            nn.ReLU())
        #图转换
        self.transform = nn.Sequential(
            Dense(self.graph_node_dim, self.graph_node_dim))
        #生成边嵌入（与原文有改变）
        self.gen_edge_emb = nn.Sequential(
            Dense(self.graph_node_dim*2, self.graph_node_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            Dense(self.graph_node_dim, self.graph_node_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            Dense(self.graph_node_dim, self.graph_node_dim)
        )
        
        self.f_gate = Dense(self.h_dim, self.h_dim)
        
    def forward(self, x):
        x_res = x.clone()
        time_step, batch_size, _ = x_res.size()
        #3.5 Node Embedding of DGP,没有经过SRU
        ht_enc = x_res
        ht_enc = torch.reshape(ht_enc, (time_step, batch_size, self.h_dim)).permute(1, 0, 2)
        #激活函数
        f_t = torch.sigmoid(self.f_gate(ht_enc))

        h_enc = f_t * ht_enc + (1-f_t)*torch.tanh(ht_enc)
        #vi = ReLU(MLP(˜vi))
        node_emb = self.node_emb(h_enc.clone())
        #在ht_enc已经改过了
        # node_emb = node_emb.permute(1,0,2)
        #点对
        node_pairs = torch.zeros(batch_size, self.total_window_size, self.graph_node_dim * 2).cuda()

        #生成图
        for i in range(self.window_size-1):
            start = int((self.window_size-i-2)*(self.window_size-i-1)/2)
            end = int((self.window_size-i)*(self.window_size-i-1)/2)
            one = node_emb[:, self.window_size-i-1, :].unsqueeze(1).repeat(1, self.window_size-i-1, 1)
            two = node_emb[:, 0:self.window_size-i-1,:]
            node_pairs[:, start:end, :] = torch.cat([one, two],dim=2)
        
        # node2edge
        # node_pairs(21,16,256)
        # edge_emb(21,16,128)
        edge_emb = self.gen_edge_emb(node_pairs)
        input4prior = edge_emb.clone()
        input4post = edge_emb.clone()

        #Inference of Binomial Edge Variables of DGP（伯努利计算变为泊松计算)
        # prior
        prior_mean_g, prior_std_g, prior_b = self.prior_enc(input4prior)
        # estimate prior mij for Binomial Dis
        prior_mij = self.prior_mij(prior_b)
        prior_mij = 0.4 * sigmoid(prior_mij)

        # post 
        post_mean_g, post_std_g, post_b = self.post_enc(input4post)
        post_mean_approx_g = self.post_mean_approx_g(post_b)
        post_std_approx_g = self.post_std_approx_g(post_b)

        # estimate post mij for Binomial Dis
        '''Poisson'''
        eps = 1e-6
        nij = 2.0 * post_mean_approx_g - 1.0
        nij_ = nij.pow(2) + 8.0 * post_std_approx_g.pow(2)
        post_mij = 0.25 * (nij + torch.sqrt(nij_)) + eps
        '''
        nij = softplus(post_mean_approx_g) + 0.01
        nij_ = 2.0 * nij * post_std_approx_g.pow(2)
        post_mij = 0.5 * (1.0 + nij_ - torch.sqrt(nij_.pow(2) + 1))
        '''
        #alpha nar是经过S的图变换之后的图，alpha_tilda是summary graph
        # reparameterization: sampling alpha_tilda and alpha_bar
        alpha_bar, alpha_tilde = self.sample_repara(post_mean_g, post_std_g, post_mij)
        alpha_bar = torch.relu(alpha_bar)
        # graph embedding
        ei = torch.sum(torch.mul(alpha_bar, edge_emb), dim=1) / (torch.sum(alpha_bar, dim=1)+1e-6)
        transformed_ei = self.transform(ei).unsqueeze(0).repeat(time_step, 1, 1)

        # skip connection
        if self.res:
            x_res_ei = torch.cat([x_res, transformed_ei], dim=2)
            ht_res, _ = self.sru_res(x_res_ei, None)
        else:
            ht_res, _ = self.sru_res(transformed_ei, None)

        # regularization
        kl_g = self.kld_loss_gauss(alpha_tilde * post_mean_g, torch.sqrt(alpha_tilde) * post_std_g,
                                   alpha_tilde * prior_mean_g, torch.sqrt(alpha_tilde) * prior_std_g)
        #(ELBO)        
        # #公式15、16
        kl_b = self.kld_loss_binomial_upper_bound(post_mij, prior_mij)

        # print("ht_res = ",ht_res.size())
        # return dic for next iter and optimization
        result_dic = {
            'ht_output': ht_res,
            'kl_g': kl_g,
            'kl_b': kl_b,
            'summ_graph': alpha_tilde,
            'spec_graph': alpha_bar
        }
        return result_dic

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def sample_repara(self, mean, std, mij):
        mean_alpha = mij
        '''Poisson'''
        std_alpha = torch.sqrt(mij)
        '''
        std_alpha = torch.sqrt(mij*(1.0 - mij))
        '''
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        alpha_tilde = eps * std_alpha + mean_alpha
        alpha_tilde = softplus(alpha_tilde)

        mean_sij = alpha_tilde * mean
        std_sij = torch.sqrt(alpha_tilde) * std
        eps_2 = torch.FloatTensor(std.size()).normal_().cuda()
        s_ij = eps_2 * std_sij + mean_sij
        alpha_bar = s_ij * alpha_tilde

        return alpha_bar, alpha_tilde

    def kld_loss_gauss(self, mean_post, std_post, mean_prior, std_prior):
        eps = 1e-6
        kld_element = (2 * torch.log(std_prior + eps) - 2 * torch.log(std_post + eps) +
                       ((std_post).pow(2) + (mean_post - mean_prior).pow(2)) /
                       (std_prior + eps).pow(2) - 1)
        return 0.5 * torch.sum(torch.abs(kld_element))

    def kld_loss_binomial_upper_bound(self, mij_post, mij_prior):
        eps = 1e-6
        '''Poisson'''
        kld_element_term1 = mij_prior - mij_post + \
                            mij_post * (torch.log(mij_post+eps) - torch.log(mij_prior+eps))
        '''
        first_item = mij_post*(torch.log(mij_post+eps)-torch.log(mij_prior+eps))
        second_item = (1-mij_post)*(torch.log(1-mij_post+0.5*mij_post.pow(2)+eps)-
                                    torch.log(1-mij_prior+0.5*mij_prior.pow(2)+eps))
        kld_element_term1 = first_item + second_item
        '''
        return torch.sum(torch.abs(kld_element_term1))
