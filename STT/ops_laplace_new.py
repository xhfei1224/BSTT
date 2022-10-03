# 2021.7.28 Poisson-RTN

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import init
#from sru import SRU
import math
import numpy as np

from npn import *
from gcn import *

import warnings

warnings.filterwarnings('ignore')

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


########################################################################
########################################################################
########################################################################


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))    # torch.Size([2, 5])
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)    
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)     
    return true_dist

class one_hot_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(one_hot_CrossEntropy,self).__init__()
    
    def forward(self, x ,y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        loss = y*torch.log(P_i + 0.00000001)
        loss = -torch.mean(torch.sum(loss,dim=1),dim = 0)
        return loss


#transformer

class LayerNorm(nn.Module):
    """layernorm"""

    def __init__(self, features, eps=1e-6):
        #features:d_model

        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))# 类似BN层原理，对归一化后的结果进行线性偏移，feature=d_model，相当于每个embedding维度偏移不同的倍数
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        "Norm"

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
    
    "FFN"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# Positional Encoding
class PositionalEncoding(nn.Module):
    "PE"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = (torch.sin(position * div_term))[:, 0:(pe[:, 0::2].shape[-1])]    # 偶数列
        pe[:, 1::2] = (torch.cos(position * div_term))[:, 0:(pe[:, 1::2].shape[-1])]    # 奇数列
        pe = pe.unsqueeze(0)           # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_name_par(model):
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())

# Attention
def attention(query, key, value, mask=None, dropout=None):
    "Attention"
    d_k = query.size(-1)
    # [B, h, L, L]
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
 

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        MultiHeadedAttention。
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) [batch, L, d_model] ->[batch, h, L, d_model/h]
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]

        # 2)qkv :[batch, h, L, d_model/h] -->x:[b, h, L, d_model/h], attn[b, h, L, L]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        #
        return self.linears[-1](x)