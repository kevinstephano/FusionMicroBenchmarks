import os
import subprocess
import torch
from torch.nn import Module

import runner

class Fusion(Module):
    def __init__(self, norm_size):
        super(Fusion, self).__init__()
        self.bn = torch.nn.BatchNorm2d(norm_size)

    def forward(self, inputs):
        out1 = self.bn(inputs)
        out2 = torch.relu(out1)
        return out2


if __name__ == "__main__" :
    # Executes Batch Size: 1-128 by powers of 2 for Resnext 101 Batch Norm sizes
    tests = [[[0, 7, 1, 'pow'], [64, 64, 1, 'add'], [112, 112, 1, 'add'], [112, 112, 1, 'add']],
             [[0, 7, 1, 'pow'], [128, 128, 1, 'add'], [56, 56, 1, 'add'], [56, 56, 1, 'add']],
             [[0, 7, 1, 'pow'], [256, 256, 1, 'add'], [56, 56, 1, 'add'], [56, 56, 1, 'add']],
             [[0, 7, 1, 'pow'], [128, 128, 1, 'add'], [56, 56, 1, 'add'], [56, 56, 1, 'add']],
             [[0, 7, 1, 'pow'], [256, 256, 1, 'add'], [28, 28, 1, 'add'], [28, 28, 1, 'add']],
             [[0, 7, 1, 'pow'], [512, 512, 1, 'add'], [28, 28, 1, 'add'], [28, 28, 1, 'add']],
             [[0, 7, 1, 'pow'], [512, 512, 1, 'add'], [14, 14, 1, 'add'], [14, 14, 1, 'add']],
             [[0, 7, 1, 'pow'], [1024, 1024, 1, 'add'], [14, 14, 1, 'add'], [14, 14, 1, 'add']],
             [[0, 7, 1, 'pow'], [1024, 1024, 1, 'add'], [7, 7, 1, 'add'], [7, 7, 1, 'add']],
             [[0, 7, 1, 'pow'], [2048, 2048, 1, 'add'], [7, 7, 1, 'add'], [7, 7, 1, 'add']],
            ]
    op_modules = [Fusion]

    runner.runner(runner.args, op_modules, tests,  reduction_dim=1)
