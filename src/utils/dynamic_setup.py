import numpy as np
import torch
import pyopencl as cl

def create_input_setup_fn(input_args, backend):
    def input_setup(runner, matrix_dim=4096, ctx=None):
        args = []
        if backend == "cuda":
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
                    args.append(np.int32(arg["value"]))
            runner.matrix_dim = matrix_dim
            return args, None

        elif backend == "opencl":
            mf = cl.mem_flags
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
                    args.append(np.int32(arg["value"]))
            runner.matrix_dim = matrix_dim
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
                        return False
                print("Result verification passed for sampled elements")
                return True
            elif backend == "opencl":
                cl.enqueue_copy(runner.queue, runner.c_host, runner.c_buf)
                for i, j in indices:
                    if abs(runner.c_host[i, j] - expected) > tol:
                        print(f"Verification failed at [{i},{j}]: Expected {expected}, got {runner.c_host[i,j]}")
                        return False
                print("Result verification passed for sampled elements")
                return True
        # Add more verification types as needed
        raise NotImplementedError("Verification type not implemented")
    return verify