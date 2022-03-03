import os
import subprocess
import torch
from torch.nn import Module

import aot_autograd_runner

class ATen_LayerNorm(Module):
    def __init__(self, norm_size):
        super(ATen_LayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(norm_size, eps=1e-12)

    def forward(self, inputs):
        out1 = self.layer_norm(inputs)
        return out1

class BertLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u)
        s = s * s
        s = s.mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

class Fusion(Module):
    def __init__(self, norm_size):
        super(Fusion, self).__init__()
        self.layer_norm = BertLayerNorm(norm_size, eps=1e-12)

    def forward(self, inputs):
        out1 = self.layer_norm(inputs)
        return out1

if __name__ == "__main__" :
    tests = [
             [[1, 7, 1, 'add'], [128, 128, 128, 'add'], [10, 10, 1, 'pow']],
             [[8, 256, 8, 'add'], [128, 128, 128, 'add'], [10, 10, 1, 'pow']],
             [[1, 7, 1, 'add'], [1024, 1024, 128, 'add'], [11, 11, 1, 'pow']],
             [[8, 256, 8, 'add'], [1024, 1024, 128, 'add'], [11, 11, 1, 'pow']],
             [[1, 16, 1, 'add'], [2048, 2048, 128, 'add'], [14, 14, 1, 'pow']],
             [[1, 8, 1, 'add'], [2048, 2048, 128, 'add'], [25600, 25600, 1, 'add']],
            ]
    op_modules = [Fusion, ATen_LayerNorm]

    aot_autograd_runner.runner(aot_autograd_runner.args, op_modules, tests)
