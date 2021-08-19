import torch
import torch.nn.functional as F
import argparse

# Enable NVFuser
torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)


parser = argparse.ArgumentParser(description='Fusion Benchmark Runner')
parser.add_argument('--warmup-trials', default=5, type=int, help='Number of trials to not measure.')
parser.add_argument('--trials', default=100, type=int, help='Number of trials to average execution time over.')

args = parser.parse_args()

import layer_norm

def clear_l2_cache() :
    t0 = torch.empty(1024*1024*10, dtype=torch.float, device='cuda', requires_grad=False)
    t1 = t0.clone()

def gen_3d_args (seqs, hidden_dim, low, high, step) :
    name_list = []
    input_list = []
    is_bool_list = []
    grad_list = []
    for idx in range(low, high+step, step) :
        name_list.append("_sq" + str(seqs) + "_dm" + str(hidden_dim) + "_ln" + str(idx))
        input_list.append([[idx, seqs, hidden_dim]])
        is_bool_list.append([False])
        grad_list.append([[idx, seqs, hidden_dim]])
    return name_list,input_list,is_bool_list,grad_list

def gen_2x_3d_args (seqs, hidden_dim, low, high, step) :
    name_list = []
    input_list = []
    is_bool_list = []
    grad_list = []
    for idx in range(low, high+step, step) :
        name_list.append("_sq" + str(seqs) + "_dm" + str(hidden_dim) + "_ln" + str(idx))
        input_list.append([[idx, seqs, hidden_dim], [idx, seqs, hidden_dim]])
        is_bool_list.append([False, False])
        grad_list.append([[idx, seqs, hidden_dim]])
    return name_list,input_list,is_bool_list,grad_list

tests = [[bias_gelu, gen_3d_args(64, 4096, 32, 128, 32)],
         [bias_gelu, gen_3d_args(8, 4096, 32, 512, 32)],
         [div_mask_softmax_dropout, gen_4d_args(64, 16, 32, 128, 32)],
         [div_mask_softmax_dropout, gen_4d_args(8, 16, 32, 512, 32)],
         [bias_dropout_add_layernorm, gen_2x_3d_args(64, 1024, 32, 128, 32)],
         [bias_dropout_add_layernorm, gen_2x_3d_args(8, 1024, 32, 512, 32)],
         [bias_dropout_add_layernorm_3linears, gen_2x_3d_args(64, 1024, 32, 128, 32)],
         [bias_dropout_add_layernorm_3linears, gen_2x_3d_args(8, 1024, 32, 512, 32)],
         [multihead_attention, gen_mha_args(64, 1024, 32, 128, 32)],
         [multihead_attention, gen_mha_args(8, 1024, 32, 512, 32)]]

for test_class,tensor_gen in tests :
    model = test_class.Fusion(test_class.BertConfig())
    model.cuda()
    torch.cuda.manual_seed(111)

    jit_model = torch.jit.script(model)

    name_list,input_list,is_bool_list,grad_list = tensor_gen 
    for idx in range(0, len(name_list)) :
        test_name = test_class.__name__ + "_" + name_list[idx]
        #print(test_name)
        inputs = []
        grads = []

        for input_idx in range(0, len(input_list[idx])) :
            if is_bool_list[idx][input_idx] :
                tmp = torch.randn(*input_list[idx][input_idx], device="cuda", dtype=torch.float, requires_grad=False)
                tmp_bool = tmp > 0.
                inputs.append(tmp_bool)
            else :
                inputs.append(torch.randn(*input_list[idx][input_idx], device="cuda", dtype=torch.float, requires_grad=True))

        for grad_idx in range(0, len(grad_list[idx])) :
            grads.append(torch.randn(*grad_list[idx][grad_idx], device="cuda", dtype=torch.float, requires_grad=False))
            
        start_evt_fwd = torch.cuda.Event(enable_timing=True)
        stop_evt_fwd = torch.cuda.Event(enable_timing=True)
        start_evt_bwd = torch.cuda.Event(enable_timing=True)
        stop_evt_bwd = torch.cuda.Event(enable_timing=True)
  
        elapsed_time_fwd = 0.0
        elapsed_time_bwd = 0.0
        jit_elapsed_time_fwd = 0.0
        jit_elapsed_time_bwd = 0.0
  
        for cnt in range(0, args.trials + args.warmup_trials) :
            for input_idx in range(0, len(inputs)) :
                if not is_bool_list[idx][input_idx] :
                    inputs[input_idx].grad = None
            model.zero_grad(set_to_none=True)
            clear_l2_cache()
            torch.cuda.synchronize()
            if cnt >= args.warmup_trials :
                start_evt_fwd.record()
            out = model(*inputs)
            if cnt >= args.warmup_trials :
                stop_evt_fwd.record()
                start_evt_bwd.record()
            out.backward(*grads)
            if cnt >= args.warmup_trials :
                stop_evt_bwd.record()
                stop_evt_bwd.synchronize()
                elapsed_time_fwd += start_evt_fwd.elapsed_time(stop_evt_fwd)
                elapsed_time_bwd += start_evt_bwd.elapsed_time(stop_evt_bwd)
  
        for cnt in range(0, args.trials + args.warmup_trials) :
            for input_idx in range(0, len(inputs)) :
                if not is_bool_list[idx][input_idx] :
                    inputs[input_idx].grad = None
            jit_model.zero_grad(set_to_none=True)
            clear_l2_cache()
            torch.cuda.synchronize()
            if cnt >= args.warmup_trials :
                start_evt_fwd.record()
            jit_out = jit_model(*inputs)
            if cnt >= args.warmup_trials :
                stop_evt_fwd.record()
                start_evt_bwd.record()
            jit_out.backward(*grads)
            if cnt >= args.warmup_trials :
                stop_evt_bwd.record()
                stop_evt_bwd.synchronize()
                jit_elapsed_time_fwd += start_evt_fwd.elapsed_time(stop_evt_fwd)
                jit_elapsed_time_bwd += start_evt_bwd.elapsed_time(stop_evt_bwd)
 
        eager_fwd_time = elapsed_time_fwd / args.trials
        jit_fwd_time = jit_elapsed_time_fwd / args.trials
        eager_bwd_time = elapsed_time_bwd / args.trials
        jit_bwd_time = jit_elapsed_time_bwd / args.trials
        per_diff_fwd = (eager_fwd_time - jit_fwd_time) / eager_fwd_time * 100.0
        per_diff_bwd = (eager_bwd_time - jit_bwd_time) / eager_bwd_time * 100.0
        print(test_name, "EAGER-Fwd:", eager_fwd_time, "Bwd:", eager_bwd_time, "JIT-Fwd:", jit_fwd_time, "Bwd:", jit_bwd_time, "%Diff_Fwd", per_diff_fwd, "%Diff_Bwd:", per_diff_bwd )
