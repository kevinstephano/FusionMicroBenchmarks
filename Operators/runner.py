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

from layer_norm import Fusion
from layer_norm import ApexLayerNorm
from layer_norm import ApexFastLayerNorm

def clear_l2_cache() :
    t0 = torch.empty(1024*1024*10, dtype=torch.float, device='cuda', requires_grad=False)
    t1 = t0.clone()

def gen_tensor_dims(recipe, partial, idx, result) :
    print("RECIPE", recipe, partial, idx)
    if idx < 0 :
        result.append(partial.reverse())
        print(partial, result)
    else :
        assert len(recipe[idx]) == 4, "All tensor dimension recipes should be 4 dimensions."
        low,high,inc,inc_type = recipe[idx]
        print("DIM RECIPE", low, high, inc, inc_type)
        assert low <= high, "Invalid recipe range."
        dim_recipe = recipe[idx]
        if inc_type == 'pow' :
            print("POW!")
            while low <= high :
                print("GEN LOW", low)
                partial.append(pow(inc, low))
                print("GEN LOW", low, inc)
                gen_tensor_dims(recipe, partial, idx-1, result)
                partial.pop()
                low += 1
        else :
            assert False, "Unrecognized dimension recipe increment type: {}".format(inc_type)

tests = [[[8, 8, 2, 'pow'], [7, 7, 2, 'pow'], [10, 10, 2, 'pow']],
        ]
impls = [Fusion, ApexLayerNorm, ApexFastLayerNorm]

for test in tests :
    print(test)
    tensor_dims = []
    gen_tensor_dims(test, [], len(test)-1, tensor_dims)
    print(tensor_dims)
    for td in tensor_dims :
        print(test, td)

"""
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
"""
