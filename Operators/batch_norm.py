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
        out1 = inputs * 1.0001
        out2 = self.bn(out1)
        return out2


if __name__ == "__main__" :
    tests = [[[256, 256, 1, 'add'], [64, 64, 1, 'add'], [112, 112, 1, 'add'], [112, 112, 1, 'add']],
             [[256, 256, 1, 'add'], [64, 64, 1, 'add'], [56, 56, 1, 'add'], [56, 56, 1, 'add']],
             [[256, 256, 1, 'add'], [256, 256, 1, 'add'], [56, 56, 1, 'add'], [56, 56, 1, 'add']],
             [[256, 256, 1, 'add'], [128, 128, 1, 'add'], [56, 56, 1, 'add'], [56, 56, 1, 'add']],
             [[256, 256, 1, 'add'], [128, 128, 1, 'add'], [28, 28, 1, 'add'], [28, 28, 1, 'add']],
             [[256, 256, 1, 'add'], [512, 512, 1, 'add'], [28, 28, 1, 'add'], [28, 28, 1, 'add']],
             [[256, 256, 1, 'add'], [256, 256, 1, 'add'], [28, 28, 1, 'add'], [28, 28, 1, 'add']],
             [[256, 256, 1, 'add'], [256, 256, 1, 'add'], [14, 14, 1, 'add'], [14, 14, 1, 'add']],
             [[256, 256, 1, 'add'], [1024, 1024, 1, 'add'], [14, 14, 1, 'add'], [14, 14, 1, 'add']],
             [[256, 256, 1, 'add'], [512, 512, 1, 'add'], [14, 14, 1, 'add'], [14, 14, 1, 'add']],
             [[256, 256, 1, 'add'], [512, 512, 1, 'add'], [7, 7, 1, 'add'], [7, 7, 1, 'add']],
             [[256, 256, 1, 'add'], [2048, 2048, 1, 'add'], [7, 7, 1, 'add'], [7, 7, 1, 'add']],
            ]
    #tests = [[[8, 8, 1, 'pow'], [128, 128, 128, 'add'], [10, 10, 1, 'pow']],
    #        ]
    op_modules = [Fusion]

    runner.runner(runner.args, op_modules, tests,  reduction_dim=1)
