import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
import math
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_features, out_features, num_node=None, bias=True, input_vector=False):
        super(GCN, self).__init__()
        #print("in_features:")
        #print(in_features)
        #print(out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.input_vector = input_vector
        self.num_node = num_node
        
        self.weight = Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        print('GCN reset')
        if (self.in_features == self.out_features):
            init.orthogonal_(self.weight)
            #print(self.weight.shape)
        else:
            init.uniform_(self.weight,
                          a=-math.sqrt(1.0 / self.in_features) * math.sqrt(3),
                          b=math.sqrt(1.0 / self.in_features) * math.sqrt(3))
        if self.bias is not None:
            init.uniform_(self.bias, -0, 0)

    def forward(self, X, A):
        '''
        A: [N,V,V] or [V,V]
        X: [N,V,F]
        '''
        if self.input_vector:
            A = self.vector2matrix(A)
            #print(A.shape, X.shape, self.weight.shape, self.bias.shape)
            return torch.matmul(torch.matmul(A, X), self.weight) + self.bias, A
        else:
            return torch.matmul(torch.matmul(A, X), self.weight) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
    
    def vector2matrix(self, A):
        if self.num_node is None:
            num_node = int(np.sqrt(1+A.shape[-1]*8)+1)//2
        else:
            num_node = self.num_node
        if(len(A.shape)==2):
            A_M = torch.zeros(A.shape[0], num_node, num_node).cuda()
            eid = 0
            for i in range(num_node):
                for j in range(i):
                    A_M[:,i,j] = A[:,eid]
                    A_M[:,j,i] = A[:,eid]
                    eid+=1
        else:
            A_M = torch.zeros(num_node, num_node).cuda()
            eid = 0
            for i in range(num_node):
                for j in range(i):
                    A_M[i,j] = A[eid]
                    A_M[j,i] = A[eid]
                    eid+=1
        A_M += torch.eye(num_node).cuda()
        return A_M