from numba import njit, cuda
import numpy as np


# --- 辅助函数：用于 CPU JIT 环境 ---
@njit
def my_func_cpu(args_sequence):
    """
    CPU JIT 函数：接收一个序列，并尝试访问其元素。
    ***演示在越界访问时，Numba @njit 也会读取未定义值（类似 GPU）***
    """
    print(f"CPU JIT: Inside my_func_cpu, args_sequence received: {args_sequence}")

    # 场景 A: 越界访问
    # 假设我们传入的是 2 个元素的序列 [20, 30] (来自 my_array[1:3])
    # 但我们尝试访问第 3 个元素 (索引 2)
    val_at_idx_2 = args_sequence[2]
    # 预期：在标准 Python 中，这里会抛出 IndexError。
    # 实际：在 Numba @njit 编译后，它像 GPU 一样，可能会读取到越界位置的内存值。
    print(
        f"CPU JIT: Accessed args_sequence[2] (potentially out-of-bounds): {val_at_idx_2}"
    )

    arg1 = args_sequence[0]
    arg2 = args_sequence[1]

    # 场景 B: 序列解包赋值
    # Numba @njit 不支持对 NumPy 数组切片进行不匹配的运行时解包，
    # 这种情况下通常会在编译时直接报错（TypingError），而不是运行时抛 ValueError。
    # 所以为了演示越界行为，我们保持直接索引访问。

    return arg1 + arg2 + val_at_idx_2


# --- 辅助设备函数：用于 GPU CUDA 环境 ---
@cuda.jit(device=True)
def my_func_gpu(args_slice):  # GPU 设备函数接收数组切片
    """
    GPU 设备函数：接收一个数组切片，并尝试访问其元素。
    演示 GPU 在越界访问时的“未定义行为”。
    """
    # 场景 A: 越界访问 (GPU 不会报错，会读到未定义值)
    val_at_idx_2 = args_slice[2]
    # 在 GPU 上，越界读取通常不会报错，而是读取内存中对应位置的值。
    # 根据之前的测试，这会读取到原始 my_array 中索引 3 的值（即 40）。

    arg1 = args_slice[0]
    arg2 = args_slice[1]
    arg3 = args_slice[2]  # 再次访问，强调越界可能返回不确定值 (根据测试结果是 40)

    return arg1 + arg2 + arg3


# --- CPU 主 JIT 函数 ---
@njit
def caller_cpu_function():
    """
    CPU JIT 环境下调用 my_func_cpu。
    """
    my_array = np.array([10, 20, 30, 40], dtype=np.int32)
    # 传入一个包含 2 个元素的切片给 my_func_cpu ([20, 30])
    # my_func_cpu 会尝试访问 args_sequence[2] (越界)
    print("\n--- CPU JIT (numba.njit) 行为 ---")
    # 移除 (type: {type(my_array[1:3])}) 部分，因为它无法在 @njit 下编译
    print(f"CPU: Calling my_func_cpu with slice {my_array[1:3]}")
    result_cpu = my_func_cpu(my_array[1:3])
    print(f"CPU JIT result: {result_cpu}")
    return result_cpu


# --- GPU 主设备函数 (由 Kernel 调用) ---
@cuda.jit(device=True)
def caller_gpu_function():
    """
    GPU CUDA 环境下调用 my_func_gpu。
    """
    my_array_gpu = cuda.local.array(4, dtype=np.int32)
    my_array_gpu[0] = 10
    my_array_gpu[1] = 20
    my_array_gpu[2] = 30
    my_array_gpu[3] = 40

    # 传入一个包含 2 个元素的切片给 my_func_gpu ([20, 30])
    # my_func_gpu 会尝试访问 args_slice[2] (越界)
    result_gpu = my_func_gpu(my_array_gpu[1:3])
    return result_gpu


# --- CUDA Kernel (入口点) ---
@cuda.jit
def main_cuda_kernel(output_array_gpu):
    idx = cuda.grid(1)
    if idx == 0:  # 只让一个线程执行，简化示例
        output_array_gpu[idx] = caller_gpu_function()


# --- 主程序执行 ---
if __name__ == "__main__":
    # --- CPU 场景 ---
    # Numba @njit 在处理数组切片越界访问时，可能不会像标准 Python 那样抛出 IndexError
    # 而是会读取到内存中的值。因此，我们直接获取结果并验证。
    print("Executing CPU JIT code (observing out-of-bounds read behavior)...")
    cpu_output = caller_cpu_function()

    # 预期结果：20 (args_sequence[0]) + 30 (args_sequence[1]) + 40 (args_sequence[2] 越界读到) = 90
    expected_cpu_result = 90
    assert cpu_output == expected_cpu_result, (
        f"CPU JIT result mismatch! Expected {expected_cpu_result}, got {cpu_output}"
    )
    print(f"CPU JIT result {cpu_output} confirms out-of-bounds read (similar to GPU).")

    # --- GPU 场景 ---
    print("\n--- GPU CUDA (numba.cuda.jit) 行为 ---")
    print("Executing GPU CUDA code (observing out-of-bounds read behavior)...")

    h_output_gpu = np.zeros(1, dtype=np.int32)
    d_output_gpu = cuda.to_device(h_output_gpu)

    threads_per_block = 1
    blocks_per_grid = 1

    main_cuda_kernel[blocks_per_grid, threads_per_block](d_output_gpu)
    cuda.synchronize()  # 等待 GPU 完成

    final_gpu_result = d_output_gpu.copy_to_host()
    print(f"GPU CUDA result: {final_gpu_result[0]}")

    # 预期结果：20 (args_slice[0]) + 30 (args_slice[1]) + 40 (args_slice[2] 越界读到) = 90
    expected_gpu_result = 90
    assert final_gpu_result[0] == expected_gpu_result, (
        f"GPU result mismatch! Expected {expected_gpu_result}, got {final_gpu_result[0]}"
    )
    print("GPU CUDA result confirms out-of-bounds read (similar to CPU JIT).")

    print("\n--- 总结 ---")
    print(
        "**重要发现：** 在 Numba 中，无论是 CPU JIT 模式还是 GPU CUDA 模式，对传入的 NumPy 数组切片进行索引越界访问时，都不会抛出 Python 标准的 `IndexError`。"
    )
    print(
        "相反，两者都表现为**未定义行为**，即会尝试读取越界位置的内存内容，这可能导致不可预测或不正确的结果（在本例中读取到的是原始数组中紧邻的值）。"
    )
    print(
        "因此，在 Numba 编译的代码中（无论是 CPU 还是 GPU），**始终需要通过逻辑判断或显式边界检查来确保数组访问在有效范围内**。"
    )
