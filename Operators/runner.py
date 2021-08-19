import argparse
from functools import reduce
import operator
from operator import itemgetter

import torch
import torch.nn.functional as F

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
parser.add_argument('--fp16', default=False, action='store_true', help='FP16 Precision.')
parser.add_argument('--inference', default=False, action='store_true', help='Measure inference.')

args = parser.parse_args()

from layer_norm import Fusion
from layer_norm import ApexLayerNorm
from layer_norm import ApexFastLayerNorm

def clear_l2_cache() :
    t0 = torch.empty(1024*1024*50, dtype=torch.float, device='cuda', requires_grad=False)
    t1 = t0.clone()

def gen_tensor_dims(recipe) :
    if len(recipe) == 0 :
        return

    queue = []
    queue.append((0, []))

    while len(queue) > 0 :
        idx, result = queue.pop(0)
        low,high,inc,inc_type = recipe[idx]
        assert low <= high
        for val in range(low, high+inc, inc) :
            dim = val
            if inc_type == 'pow' :
               dim = pow(2, val)
            result.append(dim)
            if idx == (len(recipe)-1) :
                yield result.copy()
            else :
                queue.append((idx+1, result.copy()))
            result.pop()

tests = [[[0, 8, 1, 'pow'], [128, 512, 128, 'add'], [10, 10, 1, 'pow']],
        ]
#tests = [[[6, 6, 1, 'pow'], [128, 128, 128, 'add'], [10, 10, 1, 'pow']],
#        ]
op_modules = [Fusion, ApexLayerNorm, ApexFastLayerNorm]

# Keep runs consistent
torch.cuda.manual_seed(111)
data_type = torch.float16 if args.fp16 else torch.float32

op_impls = []
for mod in op_modules :
    if mod == Fusion :
        op_impls.append(('Eager', mod))
        op_impls.append(('NVFuser', mod))
    else :
        op_impls.append((mod.__name__, mod))
            
start_evt_fwd = torch.cuda.Event(enable_timing=True)
stop_evt_fwd = torch.cuda.Event(enable_timing=True)

start_evt_bwd = None
stop_evt_bwd = None
if not args.inference :
    start_evt_bwd = torch.cuda.Event(enable_timing=True)
    stop_evt_bwd = torch.cuda.Event(enable_timing=True)

for test in tests :
    experiments = [(td, reduce(operator.mul, td)) for td in gen_tensor_dims(test)]
    experiments = sorted(experiments, key=itemgetter(1))
    for dims,elems in experiments :
        result = "infer;" if args.inference else "train;"
        result += str(dims) + ';' + str(elems)

        # Setup Data Tensors
        inputs = torch.randn(*dims, device="cuda", dtype=data_type, requires_grad=(not args.inference))
        grads = None
        if not args.inference :
            grads = torch.randn(*dims, device="cuda", dtype=data_type, requires_grad=False)

        # Loop over model implemenatations
        for impl in op_impls :
            if impl[0] == 'NVFuser' :
                model = torch.jit.script(impl[1](dims[-1]))
            else :
                model = impl[1](dims[-1])

            if args.fp16 :
                model.half()
            model.cuda()

            elapsed_time_fwd = 0.0
            elapsed_time_bwd = 0.0
            for cnt in range(0, args.trials + args.warmup_trials) :
                # Setup Step
                if not args.inference :
                    inputs.grad = None
                    model.zero_grad(set_to_none=True)
                clear_l2_cache()
                torch.cuda.synchronize()

                # Time forward
                if cnt >= args.warmup_trials :
                    start_evt_fwd.record()
                out = model(inputs)
                if cnt >= args.warmup_trials :
                    stop_evt_fwd.record()

                # Time backward (if enabled)
                if not args.inference :
                    if cnt >= args.warmup_trials :
                        start_evt_bwd.record()
                    out.backward(grads)
                    if cnt >= args.warmup_trials :
                        stop_evt_bwd.record()

                if cnt >= args.warmup_trials :
                    torch.cuda.synchronize()
                    #stop_evt_fwd.synchronize()
                    elapsed_time_fwd += start_evt_fwd.elapsed_time(stop_evt_fwd)
                    if not args.inference :
                        #stop_evt_bwd.synchronize()
                        elapsed_time_bwd += start_evt_bwd.elapsed_time(stop_evt_bwd)
        
            fwd_time = elapsed_time_fwd / args.trials
            bwd_time = 0.0
            if not args.inference :
                bwd_time = elapsed_time_bwd / args.trials

            result += ';' + impl[0] + ';' + str(fwd_time)
            if not args.inference :
                result += ';' + str(bwd_time)

        print(result)

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
