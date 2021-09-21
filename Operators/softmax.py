import os
import subprocess
import torch
from torch.nn import Module

import runner

class Fusion(Module):
    def __init__(self, norm_size):
        super(Fusion, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        out1 = inputs * 1.0
        out2 = self.softmax(out1)
        return out2

if __name__ == "__main__" :
    tests = [[[0, 8, 1, 'pow'], [128, 1024, 128, 'add']],
            ]

    op_modules = [Fusion]

    runner.runner(runner.args, op_modules, tests)
