import torch


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
        super(SoftmaxReluClassifier, self).__init__()
        ...

    def forward(self, x):
        ...