import os
import subprocess
import torch
from torch.nn import Module

import aot_autograd_runner

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size))
        self.variance_epsilon = 1e-6

    def forward(self, x : torch.Tensor):
        variance = (x * x).mean(-1, keepdim=True)
        x_hat = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x_hat

class Fusion(Module):
    def __init__(self, norm_size):
        super(Fusion, self).__init__()
        self.rms_norm = RMSNorm(norm_size)

    def forward(self, inputs):
        out1 = self.rms_norm(inputs)
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
    op_modules = [Fusion]

    aot_autograd_runner.runner(aot_autograd_runner.args, op_modules, tests)
