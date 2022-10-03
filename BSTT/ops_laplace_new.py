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

class encode_mean_std_simple(nn.Module):
    def __init__(self, graph_node_dim, h_dim, dropout=0.1):
        super(encode_mean_std_simple, self).__init__()

        self.graph_node_dim = graph_node_dim
        self.h_dim = h_dim
        self.dropout = dropout

        self.enc = nn.Sequential(nn.Linear(graph_node_dim, h_dim),
                                 nn.ReLU(), nn.Dropout(dropout))
        #self.enc_g = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))
        self.enc_b = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))

        #self.enc_mean = nn.Linear(h_dim, h_dim)
        #self.enc_std = nn.Sequential(nn.Linear(h_dim, h_dim), nn.Softplus())

    def forward(self, x):
        enc_ = self.enc(x)
        #enc_g = self.enc_g(enc_)
        enc_b = self.enc_b(enc_)
        #mean = self.enc_mean(enc_g)
        #std = self.enc_std(enc_g)
        return enc_b

class encode_mean_std_pair(nn.Module):
    def __init__(self, graph_node_dim, h_dim, dropout=0.1):
        super(encode_mean_std_pair, self).__init__()

        self.graph_node_dim = graph_node_dim
        self.h_dim = h_dim
        self.dropout = dropout

        self.enc = nn.Sequential(nn.Linear(graph_node_dim, h_dim),
                                 nn.ReLU(), nn.Dropout(dropout))
        self.enc_g = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))
        self.enc_b = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))

        self.enc_mean = nn.Linear(h_dim, 1)
        self.enc_std = nn.Sequential(nn.Linear(h_dim, 1), nn.Softplus())

    def forward(self, x):
        enc_ = self.enc(x)
        enc_g = self.enc_g(enc_)
        enc_b = self.enc_b(enc_)
        mean = self.enc_mean(enc_g)
        std = self.enc_std(enc_g)
        return mean, std, enc_b


########################################################################
########################################################################
########################################################################

class RTN(nn.Module):
    def __init__(self, h_dim, graph_node_dim, layer_num_rnn, window_size, dropout, res=False):
        super(RTN, self).__init__()
        print('RTN_Poisson_Laplace [prior,npn,softplus,b,act][no|1|no|forward|abs]')
        self.h_dim = h_dim
        self.graph_node_dim = graph_node_dim
        self.layer_num_res = layer_num_rnn
        self.dropout_rate = dropout
        self.res = res
        self.window_size = window_size
        self.total_window_size = int((window_size * (window_size - 1)) / 2)  # no self-connection
        #self.prior_graph = torch.FloatTensor(np.load('MASS-DC-eeg_vector.npy')).cuda()
        #self.prior_graph = torch.FloatTensor(np.load('MASS-DC-eeg_vector_std.npy')).cuda()

        # recurrence
        if self.res:
            self.sru_res = GCN(2 * self.h_dim, self.h_dim, num_node=19, input_vector=True)
        else:
            self.gcn1 = GCN(self.h_dim, self.h_dim//2, num_node=window_size, input_vector=True)
            self.gcn2 = GCN(self.h_dim//2, self.h_dim//2, num_node=window_size, input_vector=False)

        # prior
        self.prior_enc = encode_mean_std_pair(self.graph_node_dim, self.graph_node_dim, self.dropout_rate)
        self.prior_mij = nn.Linear(self.graph_node_dim, 1)

        # post
        self.post_enc = encode_mean_std_pair(self.graph_node_dim, self.graph_node_dim, self.dropout_rate)
        self.post_mean_approx_g = nn.Linear(self.graph_node_dim, self.graph_node_dim)
        self.post_std_approx_g = nn.Sequential(nn.Linear(self.graph_node_dim, self.graph_node_dim), nn.Softplus())

        # graph
        #self.node_emb = nn.Sequential(nn.Linear(h_dim, self.graph_node_dim), nn.ReLU())
        self.transform = nn.Sequential(
            nn.Linear(self.graph_node_dim, self.graph_node_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.graph_node_dim, 1), nn.ReLU())
        #
        self.gen_edge_emb = nn.Sequential(
            nn.Linear(self.h_dim * 2, self.graph_node_dim), nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.graph_node_dim, self.graph_node_dim), nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.graph_node_dim, self.graph_node_dim))
        
        # Laplace graph transform(2021.11) #laplace#
        self.prior_b0 = nn.Sequential(
            nn.Linear(self.graph_node_dim, 1),
            nn.Softplus()
        )
        self.npn_trans = nn.Sequential(
            NPNLinear(self.graph_node_dim, 1),
            #NPNRelu(),
            #NPNDropout(self.dropout_rate),
            #NPNLinear(self.graph_node_dim, self.graph_node_dim),
            #NPNRelu()
        )
        self.graph_trans = nn.Sequential(
            nn.Linear(self.graph_node_dim, 1),
            nn.ReLU()
        )

    def forward(self, x):
        '''
            x: (B,C,F_h) [512,6,256]
        '''
        #x_res = x.clone()
        #print("forwrd_x:")
        #print(x.shape)
        batch_size, _, _ = x.size()

        x_node_emb = x.clone() # (B,C,F_h)

        # node pairs
        # (B,W,2*F_h), W=C(C-1)/2
        node_pairs = torch.zeros(batch_size, self.total_window_size, self.h_dim * 2).cuda()
        #node_pairs:[512, (6*5/2)=15, 256*2]
        for i in range(self.window_size - 1):
            start = int((self.window_size - i - 2) * (self.window_size - i - 1) / 2)
            end = int((self.window_size - i) * (self.window_size - i - 1) / 2)
            one = x_node_emb[:, self.window_size-i-1, :].unsqueeze(1).repeat(1, self.window_size-i-1, 1)
            two = x_node_emb[:, 0:self.window_size - i - 1, :]
            node_pairs[:, start:end, :] = torch.cat([one, two], dim=2)
        # node2edge
        edge_emb_2 = self.gen_edge_emb(node_pairs) # (B,W,F_g*2)-->(B,W,F_g)
        #print(edge_emb_2.shape)
        input4prior = edge_emb_2.clone() # (B,W,F_g)
        input4post = edge_emb_2.clone()  # (B,W,F_g)

        # prior
        prior_mean_g, prior_std_g, prior_b = self.prior_enc(input4prior)  # (B,W,F_g)
        # estimate prior mij for Binomial Dis
        prior_mij = self.prior_mij(prior_b)    # (B,W,F_g)-->(B,W,1)
        prior_mij = 0.4 * sigmoid(prior_mij)   # (B,W,1)
        #b0 = self.prior_b0(prior_b)            # (B,W,1) #laplace#

        #prior_mij = prior_mij.reshape(batch_size, -1) # (B,W)
        #b0 = b0.reshape(batch_size, -1)               # (B,W) #laplace#

        # post
        post_mean_g,post_std_g,post_b = self.post_enc(input4post)                   # (B,W,F_g)
        #print(post_b.shape)
        post_mean_approx_g = self.post_mean_approx_g(post_b) # (B,W,F_g)
        post_std_approx_g = self.post_std_approx_g(post_b)   # (B,W,F_g)

        # estimate post mij for Binomial Dis 得到mij
        '''Poisson'''
        eps = 1e-6
        nij = 2.0 * post_mean_approx_g - 1.0
        nij_ = nij.pow(2) + 8.0 * post_std_approx_g.pow(2)
        post_mij = 0.25 * (nij + torch.sqrt(nij_)) + eps     # (B,W,F_g)
        
        '''
        nij = softplus(post_mean_approx_g) + 0.01
        nij_ = 2.0 * nij * post_std_approx_g.pow(2)
        post_mij = 0.5 * (1.0 + nij_ - torch.sqrt(nij_.pow(2) + 1))
        '''
        
        # reparameterization: sampling alpha_tilda and alpha_bar
        alpha_bar, alpha_tilde = self.sample_repara(post_mean_g, post_std_g, post_mij)

        #alpha_bar = self.sample_laplace(b)    # (B,W)
        bs, l, h = alpha_bar.shape
        alpha_bar = post_mij.reshape(bs*l, h)
        alpha_bar = self.graph_trans(alpha_bar)
        alpha_bar = alpha_bar.reshape(bs, l)
        alpha_bar = torch.relu(alpha_bar)     # (B,W)

        # skip connection
        if self.res:
            print('ERROR: Not implement')
        else:
            
            H, A = self.gcn1(x, alpha_bar)    # (B,C,F_h/2)
            H = self.gcn2(H, A)               # (B,C,F_h/2)
            #print(H.shape)
            #print("=" * 128)

        #print('ht ',ht_res.shape)

        # regularization
        a1 = alpha_tilde * post_mean_g
        a2 = torch.sqrt(alpha_tilde) * post_std_g
        a3 = alpha_tilde * prior_mean_g
        a4 = torch.sqrt(alpha_tilde) * prior_std_g

        kl_g, S_kl_g = self.kld_loss_gauss(a1, a2, a3, a4)
        #print(post_mij.shape)
        #print(prior_mij.shape)
        kl_b, S_kl_b = self.kld_loss_binomial_upper_bound(post_mij, prior_mij)
        # new
        #kl_l, S_kl_l = self.kld_loss_laplace(b, b0)

        #print('graph',post_mij.mean(axis=2).shape,alpha_bar.shape)
        # return dic for next iter and optimization
        
        result_dic = {
            'ht_output': H,
            #'kl_g': kl_g,
            'kl_b': kl_b,
            'kl_l': kl_g,
            'S_kl_b': S_kl_b,
            'S_kl_l': S_kl_g,
            'summ_graph': post_mij.mean(axis=2),
            'spec_graph': alpha_bar
        }
        return result_dic

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    # new
    def sample_laplace(self, b):
        
        pi = torch.tensor(np.pi)
        mu = torch.sqrt(pi/2)*b
        std_1 = torch.sqrt((4-pi)/2)*b
        eps_1 = torch.FloatTensor(b.size()).normal_().cuda()
        std_2 = std_1*eps_1+mu
        
        eps_2 = torch.FloatTensor(b.size()).normal_().cuda()
        alpha_bar = eps_2*std_2
        return alpha_bar
    
    def sample_repara(self, mean, std, mij):
        mean_alpha = mij
        '''Poisson'''
        std_alpha = torch.sqrt(mij)
        '''
        std_alpha = torch.sqrt(mij*(1.0 - mij))
        '''
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        alpha_tilde = eps * std_alpha + mean_alpha
        alpha_tilde = softplus(alpha_tilde)#*self.prior_graph

        mean_sij = alpha_tilde * mean
        std_sij = torch.sqrt(alpha_tilde) * std
        eps_2 = torch.FloatTensor(std.size()).normal_().cuda()
        s_ij = eps_2 * std_sij + mean_sij
        alpha_bar = softplus(s_ij * alpha_tilde)

        return alpha_bar, alpha_tilde

    def kld_loss_gauss(self, mean_post, std_post, mean_prior, std_prior):
        eps = 1e-6
        kld_element = (2 * torch.log(std_prior + eps) - 2 * torch.log(std_post + eps) +
                       ((std_post).pow(2) + (mean_post - mean_prior).pow(2)) /
                       (std_prior + eps).pow(2) - 1)
        return 0.5 * torch.sum(torch.abs(kld_element)), 0.5 * kld_element

    def kld_loss_binomial_upper_bound(self, mij_post, mij_prior):
        eps = 1e-6
        '''Poisson'''
        '''
        kld_element_term1 = mij_prior - mij_post + \
                            mij_post * (torch.log(mij_post+eps) - torch.log(mij_prior+eps))
        #'''
        first_item = mij_post*(torch.log(mij_post+eps)-torch.log(mij_prior+eps))
        second_item = (1-mij_post)*(torch.log(1-mij_post+0.5*mij_post.pow(2)+eps)-
                                    torch.log(1-mij_prior+0.5*mij_prior.pow(2)+eps))
        kld_element_term1 = first_item + second_item
        #'''
        #print("*"*128)
        #print(kld_element_term1.shape)
        return torch.sum(torch.abs(kld_element_term1)), kld_element_term1

    # new
    def kld_loss_laplace(self, b, b0):
        eps = 1e-6
        loss = torch.log(b0+eps) - torch.log(b+eps) + b/(b0+eps) - 1
        return torch.sum(torch.abs(loss)), loss
    
    
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
    """layernorm module"""

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