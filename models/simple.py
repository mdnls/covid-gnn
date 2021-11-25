import torch
import numpy as np
import torch.nn as nn

# This guy will do classification of h into [s, i, r] states, it is the P(h) function.
class SoftmaxReluClassifier(torch.nn.Module):
    def __init__(self, layers, batchnorm=False):
        '''
        A fully connected ReLU network followed by softmax classification.

        - Input is d_in dimensional vector
        - For l=1...k, compute layer l activation z_l as
            z_l = ReLU(W_l z_{l-1} + b_l)
        - Layer dimensions are specified by `layers` argument, W_l has dimension (d_l x d_{l-1}) and b_l has dimension dl.
        - The output is y_k = Softmax(z_k)

        :param layers: integers [d0=d_in, d1, d2, ..., dk=dout]
        :param batchnorm: if true, include batchnorm, ie. layers are z_l = ReLU(BatchNorm(Wz + b))
        '''
        super().__init__()

        self.linears = torch.nn.ModuleList([nn.Linear(layers[l],layers[l+1])] for l in range(len(layers)-1))
        if(batchnorm):
            self.batchnorm_layers = torch.nn.ModuleList([nn.BatchNorm1d(num_features=l) for l in layers])
        self.batchnorm = batchnorm
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Softmax(dim=-1)



    def forward(self, x):
        '''
        Map input x to output y_l.

        :param x: batch of inputs with shape [N, d_in]
        :return: batch of outputs with shape [N, y]
        '''

        for i in range(len(self.linears)):
            if(self.batchnorm):
                x = self.batchnorm_layers[i](x)
            x = self.relu(self.linears[i](x))

        return self.out(x)
        

class ReluFCN(torch.nn.Module):
    def __init__(self, layers, batchnorm=False):
        '''
        A fully connected ReLU network.

        - Input is d_in dimensional vector
        - For l=1...k, compute layer l activation z_l as
            z_l = ReLU(W_l z_{l-1} + b_l)
        - Layer dimensions are specified by `layers` argument, W_l has dimension (d_l x d_{l-1}) and b_l has dimension dl.

        :param layers: integers [d0=d_in, d1, d2, ..., dk=dout]
        :param batchnorm: if true, include batchnorm, ie. layers are z_l = ReLU(BatchNorm(Wz + b))
        '''
        super().__init__()
        self.linears = torch.nn.ModuleList([nn.Linear(layers[l],layers[l+1])] for l in range(len(layers)-1))
        if(batchnorm):
            self.batchnorm_layers = torch.nn.ModuleList([nn.BatchNorm1d(num_features=l) for l in layers])
        self.batchnorm = batchnorm
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        '''
        Map input x to output y_l.

        :param x: batch of inputs with shape [N, d_in]
        :return: batch of outputs with shape [N, y]
        '''

        for i in range(len(self.linears)):
            if (self.batchnorm):
                x = self.batchnorm_layers[i](x)
            x = self.relu(self.linears[i](x))
        return x

# Example of embedding function V(h)
class ClassConditionalBias(torch.nn.Module):
    def __init__(self, n_classes, dim):
        '''
        Given input x and a discrete class z, add a class-specific vector to x and output y = x + b_z

        :param n_classes: number of classes. Each class is indexed by an integer 0 <= z < n_classes.
        :param dim: dimension of x
        '''

        self.n_classes = len(self.classes)
        self.biases = torch.nn.Parameter(torch.zeros(n_classes, dim))

    def forward(self, x, classes):
        '''
        Apply class conditional bias to x.
        :param x: tensor with dimension [N, d_in]
        :param classes: tensor with dimension [N,] containing integerz 0 <= z < n_classes.
        :return: y = x+bz
        '''
        return x + torch.take(self.biases, classes)
