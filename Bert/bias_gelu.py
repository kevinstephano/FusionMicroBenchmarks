import torch
from torch.nn import Module
import torch.nn.functional as F

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

    def forward(self, inputs):
        out1 = inputs + self.bias
        out2 = F.gelu(out1)
        return out2


inputs = torch.randn(8, 128, 1024, device="cuda", dtype=torch.float, requires_grad=True)
grads = torch.randn(8, 128, 1024, device="cuda", dtype=torch.float, requires_grad=False)

model = Fusion(1024)
model.cuda()

jit_model = torch.jit.script(model)

for idx in range(5) :
    if idx == 1 :
        print(jit_model.graph_for(inputs))
    if idx == 3 :
        bwd_graph = list(
            list(jit_model.get_debug_state().execution_plans.values())[
                0].code.grad_executor_states()[0].execution_plans.values()
        )[0].graph
        print(bwd_graph)
    out = jit_model.forward(inputs)
    out.backward(grads)
