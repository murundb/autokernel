import numpy as np
import torch
import pyopencl as cl

def create_input_setup_fn(input_args, backend):
    def input_setup(runner, matrix_dim=4096, ctx=None):
        args = []
        if backend == "cuda":
            runner.matrix_dim = matrix_dim

            for arg in input_args:
                if arg["type"] == "matrix":
                    shape = tuple(arg["shape"])
                    dtype = getattr(torch, arg["dtype"])
                    if arg["init"] == "ones":
                        arr = torch.ones(shape, dtype=dtype)
                    elif arg["init"] == "zeros":
                        arr = torch.zeros(shape, dtype=dtype)
                    else:
                        raise ValueError(f"Unknown init: {arg['init']}")
                    arr = arr.cuda()
                    # Save input/output tensors on runner for later verification
                    if arg["role"] == "input":
                        setattr(runner, f"{arg['name'].lower()}_gpu", arr)
                    elif arg["role"] == "output":
                        setattr(runner, f"{arg['name'].lower()}_gpu", arr)
                    args.append(int(arr.data_ptr()))
                elif arg["type"] == "int":
                    if arg.get("role") == "output":
                        arr = torch.zeros(1, dtype=torch.int32, device="cuda")
                        setattr(runner, f"{arg['name'].lower()}_gpu", arr)
                        args.append(int(arr.data_ptr()))
                    else:
                        args.append(np.int32(arg["value"]))
                elif arg["type"] == "float":
                    if arg.get("role") == "output":
                        arr = torch.zeros(1, dtype=torch.float32, device="cuda")
                        setattr(runner, f"{arg['name'].lower()}_gpu", arr)
                        args.append(int(arr.data_ptr()))
                    else:
                        args.append(np.float32(arg["value"]))
                if arg["name"] == "matrix_dim":
                    runner.matrix_dim = arg["value"]
                    
            return args, None

        elif backend == "opencl":
            mf = cl.mem_flags
            runner.matrix_dim = matrix_dim
            for arg in input_args:
                if arg["type"] == "matrix":
                    shape = tuple(arg["shape"])
                    dtype = np.dtype(arg["dtype"])
                    if arg["init"] == "ones":
                        arr = np.ones(shape, dtype=dtype)
                    elif arg["init"] == "zeros":
                        arr = np.zeros(shape, dtype=dtype)
                    else:
                        raise ValueError(f"Unknown init: {arg['init']}")
                    if arg["role"] == "input":
                        buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr)
                        setattr(runner, f"{arg['name'].lower()}_buf", buf)
                        args.append(buf)
                    elif arg["role"] == "output":
                        buf = cl.Buffer(ctx, mf.WRITE_ONLY, arr.nbytes)
                        setattr(runner, f"{arg['name'].lower()}_buf", buf)
                        setattr(runner, f"{arg['name'].lower()}_host", arr)
                        args.append(buf)
                elif arg["type"] == "int":
                    if arg.get("role") == "output":
                        arr = np.zeros(1, dtype=np.int32)
                        buf = cl.Buffer(ctx, mf.WRITE_ONLY, arr.nbytes)
                        setattr(runner, f"{arg['name'].lower()}_buf", buf)
                        setattr(runner, f"{arg['name'].lower()}_host", arr)
                        args.append(buf)
                    else:
                        args.append(np.int32(arg["value"]))
                elif arg["type"] == "float":
                    if arg.get("role") == "output":
                        arr = np.zeros(1, dtype=np.float32)
                        buf = cl.Buffer(ctx, mf.WRITE_ONLY, arr.nbytes)
                        setattr(runner, f"{arg['name'].lower()}_buf", buf)
                        setattr(runner, f"{arg['name'].lower()}_host", arr)
                        args.append(buf)
                    else:
                        args.append(np.float32(arg["value"]))
                if arg["name"] == "matrix_dim":
                    runner.matrix_dim = arg["value"]
            return args, None
        else:
            raise ValueError(f"Unknown backend: {backend}")
    return input_setup

def create_verification_fn(verification, backend):
    def verify(runner, matrix_dim=4096):
        if verification["type"] == "matrix_equals":
            expected = float(verification["expected_value"])
            indices = verification["sample_indices"]
            tol = verification.get("tolerance", 1e-5)
            if backend == "cuda":
                c_host = runner.c_gpu.cpu()
                for i, j in indices:
                    if abs(c_host[i, j].item() - expected) > tol:
                        print(f"Verification failed at [{i},{j}]: Expected {expected}, got {c_host[i,j].item()}")
                        return False, f"Verification failed at [{i},{j}]: Expected {expected}, got {c_host[i,j].item()}"
                print("Result verification passed for sampled elements")
                return True, "Result verification passed for sampled elements"
            elif backend == "opencl":
                cl.enqueue_copy(runner.queue, runner.result_host, runner.result_buf)
                for i, j in indices:
                    if abs(runner.result_host[i, j] - expected) > tol:
                        print(f"Verification failed at [{i},{j}]: Expected {expected}, got {runner.result_host[i,j]}")
                        return False, f"Verification failed at [{i},{j}]: Expected {expected}, got {runner.result_host[i,j]}"
                print("Result verification passed for sampled elements")
                return True, "Result verification passed for sampled elements"
        elif verification["type"] == "scalar_equals":
            expected = float(verification["expected_value"])
            tol = verification.get("tolerance", 1e-5)
            if backend == "cuda":
                # Assume scalar is stored in runner.c_gpu as a 1-element tensor
                c_host = runner.result_gpu.cpu().item() if hasattr(runner.result_gpu, "cpu") else runner.result_gpu
                if abs(c_host - expected) > tol:
                    print(f"Verification failed: Expected {expected}, got {c_host}")
                    return False, f"Verification failed: Expected {expected}, got {c_host}"
                print("Result verification passed for scalar")
                return True, "Result verification passed for scalar"
            elif backend == "opencl":
                # Assume scalar is stored in runner.c_host as a numpy array of shape (1,)
                cl.enqueue_copy(runner.queue, runner.result_host, runner.result_buf)
                c_host = runner.result_host
                if hasattr(c_host, "size") and c_host.size == 1:
                    c_host_scalar = float(c_host.flat[0])
                else:
                    c_host_scalar = float(c_host)
                if abs(c_host_scalar - expected) > tol:
                    print(f"Verification failed: Expected {expected}, got {c_host_scalar}")
                    return False, f"Verification failed: Expected {expected}, got {c_host_scalar}"
                print("Result verification passed for scalar")
                return True, "Result verification passed for scalar"
        # Add more verification types as needed
        raise NotImplementedError("Verification type not implemented")
    return verify