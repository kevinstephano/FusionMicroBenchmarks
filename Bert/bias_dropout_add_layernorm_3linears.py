import torch
from torch.nn import Module

torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)

class Fusion(Module):
    def __init__(self, hidden_size):
        super(Fusion, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.dropout = torch.nn.Dropout(0.1)
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.linear_a = torch.nn.Linear(1024,1024)
        self.linear_b = torch.nn.Linear(1024,1024)
        self.linear_c = torch.nn.Linear(1024,1024)

    def forward(self, inputs1, inputs2):
        out1 = inputs1 + self.bias
        out2 = self.dropout(out1)
        out3 = out2 + inputs2
        out4 = self.layer_norm(out3)
        out5 = self.linear_a(out4)
        out6 = self.linear_b(out4)
        out7 = self.linear_c(out4)
        return out5,out6,out7

inputs1 = torch.randn(256, 128, 1024, device="cuda", dtype=torch.float, requires_grad=True)
inputs2 = torch.randn(256, 128, 1024, device="cuda", dtype=torch.float, requires_grad=True)
grads1 = torch.randn(256, 128, 1024, device="cuda", dtype=torch.float, requires_grad=False)
grads2 = torch.randn(256, 128, 1024, device="cuda", dtype=torch.float, requires_grad=False)
grads3 = torch.randn(256, 128, 1024, device="cuda", dtype=torch.float, requires_grad=False)

model = Fusion(1024)
model.cuda()

jit_model = torch.jit.script(model)

for idx in range(5) :
    if idx == 3 :
        print(jit_model.graph_for(inputs1, inputs2))
        for state in list(jit_model.get_debug_state().execution_plans.values())[0].code.grad_executor_states() :
            print(list(state.execution_plans.values())[0].graph)
    out1,out2,out3 = jit_model.forward(inputs1, inputs2)
    torch.autograd.backward((out1,out2,out3), (grads1,grads2,grads3))
