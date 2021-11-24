import torch


# This guy will do classification of h into [s, i, r] states, ie. it is the P(h) function.
class SoftmaxClassifier(torch.nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()

    def forward(self, x):
        ...