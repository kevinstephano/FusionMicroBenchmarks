import torch
from torch.nn import Module

import runner

class Fusion(Module):
    def __init__(self, dims):
        super(Fusion, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, dims[-1]))
        self.dropout = torch.nn.Dropout(0.1)
        self.layer_norm = torch.nn.LayerNorm(dims[-1], eps=1e-12)

    def forward(self, inputs1, inputs2):
        out1 = inputs1 + self.bias
        out2 = self.dropout(out1)
        out3 = out2 + inputs2
        out4 = self.layer_norm(out3)
        return out4

if __name__ == "__main__" :
    tests = [[[8, 8, 1, 'pow'], [128, 128, 128, 'add'], [10, 10, 1, 'pow']],
            ]
    op_modules = [Fusion]

    runner.runner(runner.args, op_modules, tests, 2)
