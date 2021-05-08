import torch
import torch.nn.functional as F

torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)

def func(inputs, conv1_weights, conv1_bias, conv2_weights, conv2_bias) :
    # [64, 192, 28, 28] -> [64, 192]
    out1 = inputs.mean(dim=(2,3))
    # [64, 192] -> [64, 1, 192]
    out1 = out1.unsqueeze(-1)

    # [16, 192] -> [1, 16, 192]
    conv1_weights = conv1_weights.unsqueeze(0)
    # [16] -> [1, 16]
    conv1_bias = conv1_bias.unsqueeze(0)
    # [64, 16, 192]
    conv1_mul = out1 * conv1_weights
    # [64, 16]
    conv1_red = conv1_mul.sum(dim=-1)
    bias1 = conv1_red + conv1_bias

    silu_out = F.silu(bias1)
    # [64, 16] -> [64, 1, 16]
    silu_out = silu_out.unsqueeze(1)
    
    # [192, 16] -> [1, 192, 16]
    conv2_weights = conv2_weights.unsqueeze(0)
    # [192] -> [1, 192] 
    conv2_bias = conv2_bias.unsqueeze(0)
    # [64, 192, 16]
    conv2_mul = silu_out * conv2_weights
    # [64, 192]
    conv2_red = conv2_mul.sum(dim=-1)
    bias2 = conv2_red + conv2_bias

    # [64, 192, 1]
    bias2 = bias2.unsqueeze(2)
    # [64, 192, 1, 1]
    bias2 = bias2.unsqueeze(2)
    # [64, 192, 28, 28]
    scale_out = inputs * bias2
    return scale_out
"""
def func(inputs, conv1_weights, conv1_bias, conv2_weights, conv2_bias) :        
                                                                                
    # [16, 192] -> [16, 192, 1, 1]                                              
    conv1_weights = conv1_weights.unsqueeze(-1).unsqueeze(-1)                   
    # [16] -> [16, 1, 1]                                                        
    conv1_bias = conv1_bias.unsqueeze(-1).unsqueeze(-1)                         
    # [192, 16, 1, 1] -> [192, 16, 1, 1]                                        
    conv2_weights = conv2_weights.unsqueeze(-1).unsqueeze(-1)                   
    # [192, 1, 1] -> [1, 192, 1, 1]                                             
    conv2_bias = conv2_bias.unsqueeze(-1).unsqueeze(-1)                         
                                                                                
                                                                                
    # [64, 192, 28, 28] -> [64, 1, 1, 192, 28, 28, 1, 1]                        
    out1 = inputs.unsqueeze(1).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)         
    # [64, 1, 1, 192, 1, 1] -> [64, 1, 1, 192, 1, 1]                            
    out1 = out1.sum(dim=(-3,-4))                                               
    # [64, 1, 16, 192, 1, 1]                                                    
    conv1_mul = out1 * conv1_weights                                            
    # [64, 1, 16, 1, 1]                                                         
    conv1_red = conv1_mul.sum(dim=-3)                                           
    bias1 = conv1_red + conv1_bias                                              
    silu_out = F.relu(bias1)                                                    
                                                                                
    # [64, 192, 16, 1, 1]                                                       
    conv2_mul = silu_out * conv2_weights                                        
    # [64, 192, 1, 1]                                                           
    conv2_red = conv2_mul.sum(dim=-3)                                           
    bias2 = conv2_red + conv2_bias                                              
                                                                                
    scale_out = inputs * bias2                                                  
    return scale_out
"""

if __name__ == "__main__" :
    inputs = torch.randn(64, 192, 28, 28, device="cuda", dtype=torch.float, requires_grad=False)
    conv1_weights = torch.randn(16, 192, device="cuda", dtype=torch.float, requires_grad=False)
    conv1_bias = torch.randn(16, device="cuda", dtype=torch.float, requires_grad=False)
    conv2_weights = torch.randn(192, 16, device="cuda", dtype=torch.float, requires_grad=False)
    conv2_bias = torch.randn(192, device="cuda", dtype=torch.float, requires_grad=False)

    jit_model = torch.jit.script(func)

    for idx in range(4) :
        if idx == 3 :
            print(jit_model.graph_for(inputs, conv1_weights, conv1_bias, conv2_weights, conv2_bias))
        out = jit_model(inputs, conv1_weights, conv1_bias, conv2_weights, conv2_bias)
