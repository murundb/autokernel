def get_device_info(backend):
    if backend == "cuda":
        import pycuda.driver as cuda
        cuda.init()
        device = cuda.Device(0)
        return {
            "Device Name": device.name(),
            "Compute Capability": f"{device.compute_capability()}",
            "Total Memory": f"{device.total_memory() // (1024*1024)} MB",
            "Clock Rate": f"{device.clock_rate / 1000:.2f} MHz",
            "Max Threads Per Block": device.max_threads_per_block,
            "Max Block Dimensions": device.max_block_dim_x,
            "Max Grid Dimensions": device.max_grid_dim_x,
            "Warp Size": device.warp_size,
            "Max Shared Memory Per Block": f"{device.max_shared_memory_per_block // 1024} KB",
            "Max Registers Per Block": device.max_registers_per_block,
            "Async Engine Count": device.async_engine_count
        }
    else:  # OpenCL
        import pyopencl as cl
        platforms = cl.get_platforms()
        devices = platforms[0].get_devices()
        device = devices[0]
        return {
            "Device Name": device.name,
            "Device Type": cl.device_type.to_string(device.type),
            "Device Vendor": device.vendor,
            "Device Version": device.version,
            "Driver Version": device.driver_version,
            "Max Compute Units": device.max_compute_units,
            "Max Work Group Size": device.max_work_group_size,
            "Global Memory Size": f"{device.global_mem_size // (1024*1024)} MB",
            "Local Memory Size": f"{device.local_mem_size // 1024} KB",
            "Max Work Item Dimensions": device.max_work_item_dimensions,
            "Max Work Item Sizes": device.max_work_item_sizes,
            "Max Clock Frequency": f"{device.max_clock_frequency} MHz",
            "Image Support": device.image_support,
            "Extensions": device.extensions.split(),
            "Preferred Vector Width Float": device.preferred_vector_width_float,
            "Native Vector Width Float": device.native_vector_width_float
        }
