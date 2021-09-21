import os
import subprocess
import torch
from torch.nn import Module

import runner

try :
    from apex.normalization.fused_layer_norm import FusedLayerNorm
    from apex.contrib.layer_norm import FastLayerNorm
except ModuleNotFoundError :
    apex_dir = '/opt/pytorch/apex'
    assert os.path.exists(apex_dir), "Apex is not installed in expected directory: {}".format(apex_dir)
    curr_dir = os.getcwd()
    os.chdir(apex_dir)
    cmd = ['pip', 'install', '--global-option=--fast_layer_norm', '--global-option=--cuda_ext', '.']
    subprocess.run(cmd)
    os.chdir(curr_dir)
    from apex.contrib.layer_norm import FastLayerNorm

class Fusion(Module):
    def __init__(self, norm_size):
        super(Fusion, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(norm_size, eps=1e-12)

    def forward(self, inputs):
        out1 = inputs * 1.0
        out2 = self.layer_norm(out1)
        return out2

class ApexLayerNorm(Module):
    def __init__(self, norm_size):
        super(ApexLayerNorm, self).__init__()
        self.layer_norm = FusedLayerNorm(norm_size, eps=1e-12)

    def forward(self, inputs):
        out1 = self.layer_norm(inputs)
        return out1

class ApexFastLayerNorm(Module):
    def __init__(self, norm_size):
        super(ApexFastLayerNorm, self).__init__()
        self.layer_norm = FastLayerNorm(norm_size, eps=1e-12)

    def forward(self, inputs):
        out1 = self.layer_norm(inputs)
        return out1

if __name__ == "__main__" :
    tests = [[[0, 8, 1, 'pow'], [128, 512, 128, 'add'], [10, 10, 1, 'pow']],
            ]
    #tests = [[[8, 8, 1, 'pow'], [128, 128, 128, 'add'], [10, 10, 1, 'pow']],
    #        ]
    op_modules = [Fusion, ApexLayerNorm, ApexFastLayerNorm]

    runner.runner(runner.args, op_modules, tests)
