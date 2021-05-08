import torch
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

import math

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
        self.num_layers = 4

class Fusion(nn.Module):
    def __init__(self, config):
        super(Fusion, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, queries, keys, values, attention_mask):
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.bmm(queries, keys.transpose(1,2))
        attention_scores = attention_scores.view(int(attention_scores.size(0) / self.num_attention_heads),
                                                 self.num_attention_heads,
                                                 attention_scores.size(1),
                                                 attention_scores.size(2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        attention_probs = attention_probs.view(attention_probs.size(0)*attention_probs.size(1), attention_probs.size(2), attention_probs.size(3))
        context_layer = torch.bmm(attention_probs, values)

        return context_layer

if __name__ == "__main__" :
    config = BertConfig()

    sequences = 8
    seq_length = 512
    head_dim = int(config.hidden_size / config.num_attention_heads)

    queries = torch.randn(sequences*config.num_attention_heads, seq_length, head_dim, device="cuda", dtype=torch.float, requires_grad=True)
    keys = torch.randn(sequences*config.num_attention_heads, seq_length, head_dim, device="cuda", dtype=torch.float, requires_grad=True)
    values = torch.randn(sequences*config.num_attention_heads, seq_length, head_dim, device="cuda", dtype=torch.float, requires_grad=True)
    mask = torch.randn(sequences, 1, 1, seq_length, device="cuda", dtype=torch.float, requires_grad=False)
    mask_bool = mask > 0.
    grads = torch.randn(sequences*config.num_attention_heads, seq_length, head_dim, device="cuda", dtype=torch.float, requires_grad=False)
   
    model = Fusion(config)
    model.cuda()
   
    jit_model = torch.jit.script(model)

    for idx in range(5) :
        if idx == 3 :
            print(jit_model.graph_for(queries, keys, values, mask_bool))
            for state in list(jit_model.get_debug_state().execution_plans.values())[0].code.grad_executor_states() :
                print(list(state.execution_plans.values())[0].graph)
        out = jit_model.forward(queries, keys, values, mask_bool)
        out.backward(grads)
