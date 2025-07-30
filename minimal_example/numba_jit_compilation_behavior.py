import numba
from numba import jit, cuda
import numpy as np
import time

print("--- Numba 编译行为综合示例 ---")

# --- CPU JIT 模式 ---
print("\n### CPU JIT 模式 ###")


@jit(nopython=True)
def process_array_cpu(arr, out):
    """
    在CPU上处理数组，根据传入数组的维度数不同，Numba 会分别编译。
    这个函数只是一个简单的元素求和示例，用于观察编译行为。
    """
    total = 0.0
    for x in np.nditer(arr):
        total += x.item()
    out[0] = total


# 场景 1: 改变形状 (维度数不变)
print("\n--- 场景 1: CPU JIT - 改变形状 (维度数不变) ---")

# 第一次调用：2x2 二维数组
print("CPU: 第一次调用 (2x2 二维数组)")
arr_cpu_2x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
out_cpu_shape1 = np.zeros(1, dtype=np.float32)

start_time = time.perf_counter()
process_array_cpu(arr_cpu_2x2, out_cpu_shape1)
end_time = time.perf_counter()
print(f"CPU: 第一次调用耗时: {end_time - start_time:.6f} 秒")
print(f"CPU: 结果: {out_cpu_shape1[0]}")
print(f"CPU: process_array_cpu 是否已编译: {process_array_cpu.signatures is not None}")
print("-" * 20)

# 第二次调用：3x3 二维数组 (只改变形状，维度数仍为2)
print("CPU: 第二次调用 (3x3 二维数组，形状改变)")
arr_cpu_3x3 = np.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
)
out_cpu_shape2 = np.zeros(1, dtype=np.float32)

start_time = time.perf_counter()
process_array_cpu(arr_cpu_3x3, out_cpu_shape2)  # 不会重新编译
end_time = time.perf_counter()
print(f"CPU: 第二次调用耗时: {end_time - start_time:.6f} 秒 (应该非常快)")
print(f"CPU: 结果: {out_cpu_shape2[0]}")
print("-" * 20)

# 场景 2: 改变维度数
print("\n--- 场景 2: CPU JIT - 改变维度数 ---")

# 第一次调用 (为了清晰，这里重新设置函数，但实际在一个函数中，Numba会为不同签名生成不同版本)
# 假设这是该函数第一次被一维数组调用
print("CPU: 第一次调用 (4元素一维数组，维度数改变)")
arr_cpu_1d = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
out_cpu_ndim1 = np.zeros(1, dtype=np.float32)

start_time = time.perf_counter()
process_array_cpu(arr_cpu_1d, out_cpu_ndim1)  # 这次会为一维数组签名重新编译
end_time = time.perf_counter()
print(f"CPU: 第一次调用耗时: {end_time - start_time:.6f} 秒 (应该再次较慢)")
print(f"CPU: 结果: {out_cpu_ndim1[0]}")
print("-" * 20)

# --- CUDA JIT 模式 ---
print("\n### CUDA JIT 模式 ###")


@cuda.jit
def process_array_cuda(arr, out):
    """
    在GPU上处理数组，这个函数只是一个简单的元素求和示例，
    使用原子加法将所有元素累加到 out[0]。
    用于观察参数类型（包括维度数）变化时的编译行为。
    """
    idx = cuda.grid(1)
    if idx < arr.size:
        cuda.atomic.add(out, 0, arr.flat[idx])


# 辅助函数来启动CUDA核函数
def run_cuda_process(a_host, out_host):
    threadsperblock = 32
    blockspergrid = (a_host.size + threadsperblock - 1) // threadsperblock

    d_a = cuda.to_device(a_host)
    d_out = cuda.to_device(out_host)

    process_array_cuda[blockspergrid, threadsperblock](d_a, d_out)
    d_out.copy_to_host(out_host)


# 场景 1: 改变形状 (维度数不变)
print("\n--- 场景 1: CUDA JIT - 改变形状 (维度数不变) ---")

# 第一次调用：2x2 二维数组
print("CUDA: 第一次调用 (2x2 二维数组)")
arr_cuda_2x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
out_cuda_shape1 = np.zeros(1, dtype=np.float32)  # 初始化为0

start_time = time.perf_counter()
run_cuda_process(arr_cuda_2x2, out_cuda_shape1)
end_time = time.perf_counter()
print(f"CUDA: 第一次调用耗时: {end_time - start_time:.6f} 秒")
print(f"CUDA: 结果: {out_cuda_shape1[0]}")
print("-" * 20)

# 第二次调用：4x4 二维数组 (只改变形状，维度数仍为2)
print("CUDA: 第二次调用 (4x4 二维数组，形状改变)")
arr_cuda_4x4 = np.ones((4, 4), dtype=np.float32) * 5  # 示例数据
out_cuda_shape2 = np.zeros(1, dtype=np.float32)  # 初始化为0

start_time = time.perf_counter()
run_cuda_process(arr_cuda_4x4, out_cuda_shape2)  # 不会重新编译
end_time = time.perf_counter()
print(f"CUDA: 第二次调用耗时: {end_time - start_time:.6f} 秒 (应该非常快)")
print(f"CUDA: 结果: {out_cuda_shape2[0]}")
print("-" * 20)

# 场景 2: 改变维度数
print("\n--- 场景 2: CUDA JIT - 改变维度数 ---")

# 第一次调用 (为了清晰，这里重新设置函数，但实际在一个函数中，Numba会为不同签名生成不同版本)
# 假设这是该函数第一次被一维数组调用
print("CUDA: 第一次调用 (8元素一维数组，维度数改变)")
arr_cuda_1d = np.arange(1, 9, dtype=np.float32)  # 1到8的一维数组
out_cuda_ndim1 = np.zeros(1, dtype=np.float32)  # 初始化为0

start_time = time.perf_counter()
run_cuda_process(arr_cuda_1d, out_cuda_ndim1)  # 这次会为一维数组签名重新编译
end_time = time.perf_counter()
print(f"CUDA: 第一次调用耗时: {end_time - start_time:.6f} 秒 (应该再次较慢)")
print(f"CUDA: 结果: {out_cuda_ndim1[0]}")
print("-" * 20)

print("\n🎉 示例结束！")
