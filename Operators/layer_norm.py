import os
import subprocess
import torch
from torch.nn import Module

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

torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)

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
    inputs = torch.randn(8, 512, 1024, device="cuda", dtype=torch.float, requires_grad=True)
    grads = torch.randn(8, 512, 1024, device="cuda", dtype=torch.float, requires_grad=False)

    #model = Fusion(1024)
    model = FusedLayerNorm(1024)
    model.cuda()

    #jit_model = torch.jit.script(model)

    for idx in range(5) :
        #if idx == 3 :
        #    print(jit_model.graph_for(inputs))
        #    for state in list(jit_model.get_debug_state().execution_plans.values())[0].code.grad_executor_states() :
        #        print(list(state.execution_plans.values())[0].graph)
        out = model.forward(inputs)
        out.backward(grads)
