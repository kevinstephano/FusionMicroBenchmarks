import torch
from torch.nn import Module
import torch.nn.functional as F

import runner

class Fusion(Module):
    def __init__(self, dims):
        super(Fusion, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, dims[-1]))

    def forward(self, inputs):
        out1 = inputs + self.bias
        out2 = F.gelu(out1)
        return out2

if __name__ == "__main__" :
    tests = [[[8, 8, 1, 'pow'], [128, 128, 128, 'add'], [12, 12, 1, 'pow']],
            ]
    op_modules = [Fusion]

    runner.runner(runner.args, op_modules, tests, 1)
