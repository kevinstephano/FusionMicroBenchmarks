import torch
from torch.nn import Module

torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)

class BertConfig :
    def __init__(self) :
        self.hidden_size = 1024
        self.num_attention_heads = 16
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.num_layers = 10

class Fusion(Module):
    def __init__(self, config):
        super(Fusion, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(config.hidden_size))
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, inputs1, inputs2):
        out1 = inputs1 + self.bias
        out2 = self.dropout(out1)
        out3 = out2 + inputs2
        out4 = self.layer_norm(out3)
        return out4

if __name__ == "__main__" :
    inputs1 = torch.randn(8, 512, 1024, device="cuda", dtype=torch.float, requires_grad=True)
    inputs2 = torch.randn(8, 512, 1024, device="cuda", dtype=torch.float, requires_grad=True)
    grads = torch.randn(8, 512, 1024, device="cuda", dtype=torch.float, requires_grad=False)

    model = Fusion(BertConfig())
    model.cuda()

    jit_model = torch.jit.script(model)

    for idx in range(5) :
        if idx == 3 :
            print(jit_model.graph_for(inputs1, inputs2))
            for state in list(jit_model.get_debug_state().execution_plans.values())[0].code.grad_executor_states() :
                print(list(state.execution_plans.values())[0].graph)
        out = jit_model.forward(inputs1, inputs2)
        out.backward(grads)
