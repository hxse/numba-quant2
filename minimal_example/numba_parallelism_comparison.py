import numba as nb
import numpy as np
import time
import numba.cuda  # 明确导入 numba.cuda，确保其可用性

# --- CPU (njit) 并行示例 ---


@nb.njit(parallel=True)
def cpu_2_layer_parallel_complex_ops(arr, val1, val2):
    """
    一个包含两层 prange 并行，并在内层执行“不同任务”的 Numba njit 函数。
    这里的“不同任务”指在每个元素上执行不同的计算，它们互不干扰。
    例如：先加法，后乘法。
    """
    for i in nb.prange(arr.shape[0]):
        for j in nb.prange(arr.shape[1]):  # 第二层也使用 prange
            arr[i, j] += val1  # 任务 A (加法)
            arr[i, j] *= val2  # 任务 B (乘法)
    return arr


@nb.njit(parallel=True)
def cpu_1_layer_parallel_add(arr, val):
    """
    一个只有一层 prange 并行的 Numba njit 函数（推荐的 CPU 并行方式）。
    """
    for i in nb.prange(arr.shape[0]):  # 只有第一层使用 prange
        for j in range(
            arr.shape[1]
        ):  # 第二层是普通循环，由每个外层 prange 线程顺序执行
            arr[i, j] += val
    return arr


# --- CUDA (cuda.jit) 并行示例 (保持不变，因为其并行模型已经很清晰) ---


@nb.cuda.jit
def cuda_1_layer_parallel_add_kernel(arr, val):
    """
    CUDA 核函数：每个 GPU 线程处理 arr 的一行（或子范围），行内循环顺序执行。
    """
    idx_row = numba.cuda.grid(1)

    if idx_row < arr.shape[0]:
        for j in range(arr.shape[1]):
            arr[idx_row, j] += val


@nb.cuda.jit
def cuda_2_layer_parallel_add_kernel(arr, val):
    """
    CUDA 核函数：每个 GPU 线程处理 arr 的一个元素，实现二维（完全）并行。
    """
    idx_row, idx_col = numba.cuda.grid(2)

    if idx_row < arr.shape[0] and idx_col < arr.shape[1]:
        arr[idx_row, idx_col] += val


def main():
    array_size = (2000, 2000)
    add_value_cpu_1_layer = 1.0  # 用于 CPU 1层并行
    val1_cpu_2_layer = 1.0  # 用于 CPU 2层并行任务 A (加法)
    val2_cpu_2_layer = 2.0  # 用于 CPU 2层并行任务 B (乘法)
    add_value_cuda = 1.0  # 用于 CUDA

    print("--- CPU (njit) 并行测试 ---")
    data_cpu_2_layer = np.ones(array_size, dtype=np.float64)
    data_cpu_1_layer = np.ones(array_size, dtype=np.float64)

    # 运行 CPU 1层并行
    start_time = time.perf_counter()
    cpu_1_layer_parallel_add(data_cpu_1_layer, add_value_cpu_1_layer)
    end_time = time.perf_counter()
    print(f"CPU 1层并行执行时间: {end_time - start_time:.6f} 秒")
    # 初始值 1.0, 经过 +1.0 变为 2.0. 校验和 = 2.0 * 2000 * 2000 = 8,000,000.0
    print(f"结果校验和 (CPU 1层并行): {np.sum(data_cpu_1_layer)}")

    # 运行 CPU 2层并行 (执行不同任务)
    start_time = time.perf_counter()
    cpu_2_layer_parallel_complex_ops(
        data_cpu_2_layer, val1_cpu_2_layer, val2_cpu_2_layer
    )
    end_time = time.perf_counter()
    print(f"CPU 2层并行执行时间 (不同任务): {end_time - start_time:.6f} 秒")
    # 初始值 1.0, 经过 +1.0 变为 2.0, 再经过 *2.0 变为 4.0. 校验和 = 4.0 * 2000 * 2000 = 16,000,000.0
    print(f"结果校验和 (CPU 2层并行): {np.sum(data_cpu_2_layer)}")
    print("-" * 50)

    print("\n--- CUDA (cuda.jit) 并行测试 ---")
    if not numba.cuda.is_available():
        print("CUDA 不可用。跳过 CUDA 测试。")
        print("-" * 50)
        return

    try:
        device = numba.cuda.get_current_device()
        max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    except Exception as e:
        print(f"无法获取 CUDA 设备属性: {e}。使用默认 threads_per_block。")
        max_threads_per_block = 1024

    # --- GPU 1层并行 (每个线程处理一行) 配置 ---
    threads_per_block_1d = min(max_threads_per_block, array_size[0])
    blocks_per_grid_1d = (
        array_size[0] + threads_per_block_1d - 1
    ) // threads_per_block_1d

    data_cuda_1_layer = np.ones(array_size, dtype=np.float64)
    d_data_cuda_1_layer = numba.cuda.to_device(data_cuda_1_layer)

    start_time = time.perf_counter()
    cuda_1_layer_parallel_add_kernel[blocks_per_grid_1d, threads_per_block_1d](
        d_data_cuda_1_layer, add_value_cuda
    )
    numba.cuda.synchronize()
    end_time = time.perf_counter()
    h_result_cuda_1_layer = d_data_cuda_1_layer.copy_to_host()
    print(f"CUDA 1层并行执行时间 (每线程一行): {end_time - start_time:.6f} 秒")
    print(f"结果校验和 (CUDA 1层并行): {np.sum(h_result_cuda_1_layer)}")

    # --- GPU 2层并行 (每个线程处理一个元素) 配置 ---
    threads_per_block_2d_x = int(np.sqrt(max_threads_per_block))
    threads_per_block_2d_y = int(np.sqrt(max_threads_per_block))

    blocks_per_grid_2d_x = (
        array_size[0] + threads_per_block_2d_x - 1
    ) // threads_per_block_2d_x
    blocks_per_grid_2d_y = (
        array_size[1] + threads_per_block_2d_y - 1
    ) // threads_per_block_2d_y

    data_cuda_2_layer = np.ones(array_size, dtype=np.float64)
    d_data_cuda_2_layer = numba.cuda.to_device(data_cuda_2_layer)

    start_time = time.perf_counter()
    cuda_2_layer_parallel_add_kernel[
        (blocks_per_grid_2d_x, blocks_per_grid_2d_y),
        (threads_per_block_2d_x, threads_per_block_2d_y),
    ](d_data_cuda_2_layer, add_value_cuda)
    numba.cuda.synchronize()
    end_time = time.perf_counter()
    h_result_cuda_2_layer = d_data_cuda_2_layer.copy_to_host()
    print(f"CUDA 2层并行执行时间 (每线程一元素): {end_time - start_time:.6f} 秒")
    print(f"结果校验和 (CUDA 2层并行): {np.sum(h_result_cuda_2_layer)}")
    print("-" * 50)


if __name__ == "__main__":
    main()
