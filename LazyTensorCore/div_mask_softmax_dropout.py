import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

import math

import runner

class Fusion(nn.Module):
    def __init__(self, dims):
        super(Fusion, self).__init__()
        self.attention_head_size = int(dims[1] / dims[-1])
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, mask):
        out1 = inputs / math.sqrt(self.attention_head_size)
        out2 = out1 + mask
        out3 = F.softmax(out2, dim=-1)
        out4 = self.dropout(out3)
        return out4

if __name__ == "__main__" :
    tests = [[[8, 8, 1, 'pow'], [16,16,16, 'add'], [128, 128, 128, 'add'], [7, 7, 1, 'pow']],
            ]
    op_modules = [Fusion]

    runner.runner(runner.args, op_modules, tests, 2)
