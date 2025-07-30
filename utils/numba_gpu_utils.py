from numba import cuda


def get_gpu_properties():
    """
    查询当前 GPU 的硬件属性。
    """
    try:
        device = cuda.get_current_device()
        props = {
            "name": device.name,
            "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
            "warp_size": device.WARP_SIZE,
            "multiprocessor_count": device.MULTIPROCESSOR_COUNT,
            "max_registers_per_block": device.MAX_REGISTERS_PER_BLOCK,
            "max_shared_memory_per_block": device.MAX_SHARED_MEMORY_PER_BLOCK,
            "major": device.compute_capability[0],  # 添加计算能力主版本
            "minor": device.compute_capability[1],  # 添加计算能力次版本
        }
        return props
    except Exception as e:
        print(f"Error getting GPU properties: {e}")
        return {}


def calculate_optimal_threads_per_block(
    props,
    shared_mem_per_thread=0,
    max_desired_threads_per_block=512,
    register_per_thread=24,
):
    """
    根据 GPU 属性和核函数特性计算最优的 threadsperblock。
    """
    warp_size = props.get("warp_size", 32)
    max_threads_per_block = props.get("max_threads_per_block", 1024)
    max_shared_memory_per_block = props.get("max_shared_memory_per_block", 49152)
    max_registers_per_block = props.get("max_registers_per_block", 65536)

    # 启发式算法：
    # 1. 线程数是 warp_size 的倍数
    # 2. 不超过 max_threads_per_block
    # 3. 考虑寄存器限制
    optimal_threads = warp_size
    while True:
        next_threads = optimal_threads * 2
        if next_threads > max_threads_per_block:
            break
        if (next_threads * shared_mem_per_thread) > max_shared_memory_per_block:
            break
        if register_per_thread is not None:
            # Numba CUDA 编译器的寄存器分配可能与实际运行时有所不同，但这里提供一个大致的限制
            # 对于 Ampere 架构 (SM 8.0+) 每个 SM 的最大寄存器数是 65536
            # 对于 Turing 架构 (SM 7.5) 每个 SM 的最大寄存器数也是 65536
            # 所以 max_registers_per_block 通常是 65536
            # 每个线程的寄存器上限通常由设备属性中的 register_per_thread_cap 决定
            # Numba 没有直接暴露这个属性，但通常在编译时会考虑
            # 这里我们使用 Block 级别的寄存器限制来估算
            if (next_threads * register_per_thread) > max_registers_per_block:
                break
        optimal_threads = next_threads

    # 确保线程数是 warp_size 的倍数
    optimal_threads = (optimal_threads // warp_size) * warp_size
    if optimal_threads == 0:  # 避免除以零
        optimal_threads = warp_size

    # 确保线程数不超过 max_desired_threads_per_block
    optimal_threads = min(optimal_threads, max_desired_threads_per_block)

    # 调试信息：打印寄存器限制效果
    if register_per_thread is not None:
        max_threads_by_registers = max_registers_per_block // register_per_thread

    return optimal_threads


def calculate_blocks_per_grid(
    props,
    threads_per_block,
    workload_size,
    min_waves=4,
    target_blocks_per_sm_small_workload=6,
    target_blocks_per_sm_large_workload=14,
):
    """
    计算 blockspergrid，优化 GPU 占用率以提高性能。
    """
    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be greater than 0")

    # 确保覆盖整个工作负载
    required_blocks = (workload_size + threads_per_block - 1) // threads_per_block

    multi_processor_count = props.get("multiprocessor_count", 1)

    # 动态调整 target_blocks_per_sm，基于工作负载大小
    if workload_size <= 1000:
        target_blocks_per_sm = target_blocks_per_sm_small_workload
    else:
        target_blocks_per_sm = target_blocks_per_sm_large_workload

    ideal_blocks_for_occupancy = target_blocks_per_sm * multi_processor_count

    # 确保至少有 min_waves 个完整波
    min_blocks_for_waves = min_waves * multi_processor_count * target_blocks_per_sm

    # 最终的 blockspergrid 取所需块数、理想占用块数和最小波块数的最大值
    blocks_per_grid = max(
        required_blocks, ideal_blocks_for_occupancy, min_blocks_for_waves
    )

    # 确保覆盖整个工作负载的最小块数
    blocks_per_grid = max(blocks_per_grid, required_blocks)

    # 添加溢出检查，限制在 2**31 - 1
    max_grid_dim = 2**31 - 1
    blocks_per_grid = min(blocks_per_grid, max_grid_dim)

    return blocks_per_grid


def auto_tune_cuda_parameters(
    workload_size,
    max_desired_threads_per_block=512,
    register_per_thread=24,
    shared_mem_per_thread=0,
    min_waves=4,
    target_blocks_per_sm_small_workload=6,
    target_blocks_per_sm_large_workload=14,
    # 如果用户想手动指定 threadsperblock 或 blockspergrid
    manual_threadsperblock=None,
    manual_blockspergrid=None,
):
    """
    自动计算最优的 CUDA 核函数启动参数 (threadsperblock, blockspergrid)。

    用户可以通过传入可选参数来覆盖自动计算的结果。

    Args:
        workload_size (int): 工作负载的大小，例如 indicator_params.shape[0]。
        max_desired_threads_per_block (int, optional): 每个块的最大期望线程数。
                                                       默认为 512。
        register_per_thread (int, optional): 每个线程预估的寄存器使用量。
                                            默认为 24。
        shared_mem_per_thread (int, optional): 每个线程预估的共享内存使用量 (字节)。
                                               默认为 0。
        min_waves (int, optional): 每个 SM 最小的波次。默认为 4。
        target_blocks_per_sm_small_workload (int, optional): 工作负载 <= 1000 时，
                                                             每个 SM 的目标块数。默认为 6。
        target_blocks_per_sm_large_workload (int, optional): 工作负载 > 1000 时，
                                                             每个 SM 的目标块数。默认为 14。
        manual_threadsperblock (int, optional): 手动指定 threadsperblock，
                                                如果指定则覆盖自动计算结果。
        manual_blockspergrid (int, optional): 手动指定 blockspergrid，
                                              如果指定则覆盖自动计算结果。

    Returns:
        tuple: (threadsperblock, blockspergrid, max_registers_for_kernel)
                max_registers_for_kernel 等于 register_per_thread，用于 gpu_kernel_device 的 max_registers 参数。
    """
    props = get_gpu_properties()

    if manual_threadsperblock is not None:
        threadsperblock = manual_threadsperblock
    else:
        threadsperblock = calculate_optimal_threads_per_block(
            props,
            max_desired_threads_per_block=max_desired_threads_per_block,
            shared_mem_per_thread=shared_mem_per_thread,
            register_per_thread=register_per_thread,
        )

    if manual_blockspergrid is not None:
        blockspergrid = manual_blockspergrid
    else:
        blockspergrid = calculate_blocks_per_grid(
            props,
            threadsperblock,
            workload_size,
            min_waves=min_waves,
            target_blocks_per_sm_small_workload=target_blocks_per_sm_small_workload,
            target_blocks_per_sm_large_workload=target_blocks_per_sm_large_workload,
        )

    # 将 register_per_thread 作为 max_registers_for_kernel 返回
    # 这样可以将其直接传递给 gpu_kernel_device
    max_registers_for_kernel = register_per_thread

    return threadsperblock, blockspergrid, max_registers_for_kernel
