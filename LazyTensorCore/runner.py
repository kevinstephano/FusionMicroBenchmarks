import argparse
from functools import reduce
import operator
from operator import itemgetter

import torch
import torch.nn.functional as F

import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm
lazy_tensor_core._LAZYC._ltc_init_ts_backend()

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

def runner(args, op_modules, tests) :
    # Keep runs consistent
    torch.cuda.manual_seed(111)
    data_type = torch.float16 if args.fp16 else torch.float32
   
    op_impls = []
    for mod in op_modules :
        if mod.__name__ == 'Fusion' :
            op_impls.append(('Eager', mod))
            op_impls.append(('TS_NVFuser', mod))
            op_impls.append(('LTC_NVFuser', mod))
        else :
            op_impls.append((mod.__name__, mod))
     
    # Create Cuda Timing Events
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
            result += str(data_type) + ';'
            result += str(dims) + ';' + str(elems)
 
            # Loop over model implemenatations
            for impl in op_impls :
                if impl[0] == 'LTC_NVFuser' :
                    device = 'lazy'
                    model = impl[1](dims)
                elif impl[0] == 'TS_NVFuser' :
                    device = 'cuda'
                    model = torch.jit.script(impl[1](dims))
                else :
                    device = 'cuda'
                    model = impl[1](dims)

                if args.fp16 :
                    model.half()
                model.to(device=device)
            
                # Setup Data Tensors
                inputs = torch.randn(*dims, device=device, dtype=data_type, requires_grad=(not args.inference))
                grads = None
                if not args.inference :
                    grads = torch.randn(*dims, device=device, dtype=data_type, requires_grad=False)
 
                elapsed_time_fwd = 0.0
                elapsed_time_bwd = 0.0
                for cnt in range(0, args.trials + args.warmup_trials) :
                    # Setup Step
                    if not args.inference :
                        inputs.grad = None
                        model.zero_grad(set_to_none=True)
                    clear_l2_cache()
 
                    # Time forward
                    start_evt_fwd.record()
                    out = model(inputs)
                    stop_evt_fwd.record()
                    
                    if device == 'lazy' :
                        ltm.mark_step()
 
                    # Time backward (if enabled)
                    if not args.inference :
                        start_evt_bwd.record()
                        out.backward(grads)
                        stop_evt_bwd.record()
                        if device == 'lazy' :
                            ltm.mark_step()
 
                    # Collect timing results
                    if cnt >= args.warmup_trials :
                        torch.cuda.synchronize()
                        elapsed_time_fwd += start_evt_fwd.elapsed_time(stop_evt_fwd)
                        if not args.inference :
                            elapsed_time_bwd += start_evt_bwd.elapsed_time(stop_evt_bwd)

                fwd_time = elapsed_time_fwd / args.trials
                result += ';' + impl[0] + ';' + str(fwd_time)
 
                if not args.inference :
                    bwd_time = elapsed_time_bwd / args.trials
                    result += ';' + str(bwd_time)
 
            print(result)
