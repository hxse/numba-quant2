import time
from numba import njit, prange, cuda, float64
import numpy as np


# --- 辅助函数：计时器 ---
def calculate_time_wrapper(func):
    """
    一个用于测量函数执行时间的装饰器。
    """

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    return wrapper


enable_cache = False


# --- 复杂的计算逻辑（Numba CPU JIT + prange）---
@njit(parallel=True, cache=enable_cache)
def complex_cpu_calc(arr):
    """
    一个模拟复杂计算的 Numba CPU 函数。
    专注于增加编译时间（通过复杂控制流）并减少实际计算时间（通过减少迭代次数）。
    """
    rows, cols = arr.shape
    output = np.zeros_like(arr, dtype=np.float64)

    # 减少迭代次数以缩短实际计算时间，但保持内部逻辑复杂性以延长编译时间
    total_iterations_per_element = 3  # 大幅减少迭代次数

    for i in prange(rows):
        for j in prange(cols):
            val = arr[i, j]
            res = val * val + 0.000123456789

            accumulated_sum = 0.0
            temp_product = 1.0

            for iter_main in range(total_iterations_per_element):
                intermediate_val_a = (res * 0.99) + (val * iter_main * 0.001)
                intermediate_val_b = (val / 0.11) - (res *
                                                     (iter_main + 1) * 0.0005)

                if intermediate_val_a > 0.6:
                    abs_res = res if res >= 0 else -res
                    res = (res + intermediate_val_b) / (abs_res + 0.001 +
                                                        1e-12)
                    accumulated_sum += res * val
                elif intermediate_val_a < 0.2:
                    res = (res - intermediate_val_a) * (intermediate_val_b *
                                                        1.03)
                    abs_val_for_div = val if val >= 0 else -val
                    accumulated_sum -= res / (abs_val_for_div + 1e-12)
                else:
                    abs_val = val if val >= 0 else -val
                    res = res + intermediate_val_a * intermediate_val_b / (
                        abs_val + 1e-12)
                    abs_res_for_sum = res if res >= 0 else -res
                    accumulated_sum += abs_res_for_sum * 0.001

                # 减少内层循环次数
                for k in range(3):  # 减少内层循环次数
                    sub_factor = 0.5 + k * 0.02
                    res_k_calc = res * sub_factor + (val / (k + 1 + 1e-12))

                    if res_k_calc > 0.8:
                        res = res_k_calc + (res * val)**0.8  # 复杂的幂运算
                    else:
                        res = res_k_calc - (res / (val + 1e-12))**0.7  # 复杂的幂运算

                    if res > 1.5:
                        res = (res**0.123) + (val**0.5) / (res**0.05 + 1e-12)
                    elif res < -1.5:
                        abs_res_power = res if res >= 0 else -res
                        abs_val_power = val if val >= 0 else -val
                        res = -((abs_res_power)**0.123) - (
                            (abs_val_power)**0.5) * ((abs_res_power)**0.05)
                    else:
                        res = res * res * res * res * 0.01 + val / 0.987

                    accumulated_sum = accumulated_sum * 0.9 + res * 0.1
                    abs_res_prod = res if res >= 0 else -res
                    temp_product = temp_product * (1 + abs_res_prod * 1e-5)

                    abs_res_check = res if res >= 0 else -res
                    if abs_res_check < 1e-15:
                        res = 1e-15 * (1.0 if res > 0 else
                                       (-1.0 if res < 0 else 1.0))
                    elif abs_res_check > 1e18:
                        res = 1e18 * (1.0 if res > 0 else -1.0)

            output[i,
                   j] = (res + accumulated_sum / total_iterations_per_element
                         ) * temp_product * (val + 0.005) / (res - val + 1e-12)

            if np.isnan(output[i, j]) or np.isinf(output[i, j]):
                output[i, j] = 0.0

    return output


# --- 复杂的计算逻辑（Numba CUDA Kernel）---
@cuda.jit(cache=enable_cache)
def complex_gpu_kernel(arr_in, arr_out):
    """
    一个模拟复杂计算的 Numba CUDA 核函数。
    专注于增加编译时间（通过复杂控制流）并减少实际计算时间（通过减少迭代次数）。
    """
    idx, idy = cuda.grid(2)
    rows, cols = arr_in.shape

    if idx < rows and idy < cols:
        val = arr_in[idx, idy]
        res = val * val + 0.000123456789

        accumulated_sum = 0.0
        temp_product = 1.0

        total_iterations_per_element = 3  # 必须与 CPU 版本一致

        for iter_main in range(total_iterations_per_element):
            intermediate_val_a = (res * 0.99) + (val * iter_main * 0.001)
            intermediate_val_b = (val / 0.11) - (res *
                                                 (iter_main + 1) * 0.0005)

            if intermediate_val_a > 0.6:
                abs_res = res if res >= 0 else -res
                res = (res + intermediate_val_b) / (abs_res + 0.001 + 1e-12)
                accumulated_sum += res * val
            elif intermediate_val_a < 0.2:
                res = (res - intermediate_val_a) * (intermediate_val_b * 1.03)
                abs_val_for_div = val if val >= 0 else -val
                accumulated_sum -= res / (abs_val_for_div + 1e-12)
            else:
                abs_val = val if val >= 0 else -val
                res = res + intermediate_val_a * intermediate_val_b / (
                    abs_val + 1e-12)
                abs_res_for_sum = res if res >= 0 else -res
                accumulated_sum += abs_res_for_sum * 0.001

            for k in range(3):  # 必须与 CPU 版本一致
                sub_factor = 0.5 + k * 0.02
                res_k_calc = res * sub_factor + (val / (k + 1 + 1e-12))

                if res_k_calc > 0.8:
                    res = res_k_calc + (res * val)**0.8
                else:
                    res = res_k_calc - (res / (val + 1e-12))**0.7

                if res > 1.5:
                    res = (res**0.123) + (val**0.5) / (res**0.05 + 1e-12)
                elif res < -1.5:
                    abs_res_power = res if res >= 0 else -res
                    abs_val_power = val if val >= 0 else -val
                    res = -((abs_res_power)**0.123) - (
                        (abs_val_power)**0.5) * ((abs_res_power)**0.05)
                else:
                    res = res * res * res * res * 0.01 + val / 0.987

                accumulated_sum = accumulated_sum * 0.9 + res * 0.1
                abs_res_prod = res if res >= 0 else -res
                temp_product = temp_product * (1 + abs_res_prod * 1e-5)

                abs_res_check = res if res >= 0 else -res
                if abs_res_check < 1e-15:
                    res = 1e-15 * (1.0 if res > 0 else
                                   (-1.0 if res < 0 else 1.0))
                elif abs_res_check > 1e18:
                    res = 1e18 * (1.0 if res > 0 else -1.0)

        arr_out[idx,
                idy] = (res + accumulated_sum / total_iterations_per_element
                        ) * temp_product * (val + 0.005) / (res - val + 1e-12)

        if arr_out[idx, idy] != arr_out[idx, idy] or arr_out[
                idx, idy] == float('inf') or arr_out[idx,
                                                     idy] == float('-inf'):
            arr_out[idx, idy] = 0.0


# --- 主测试函数 ---
def run_tests():
    # 准备假数据和大型数据
    dummy_data = np.random.rand(2, 2).astype(np.float64) * 0.2 + 0.3
    data_size = (4000, 4000)
    large_data_cpu = np.random.rand(*data_size).astype(np.float64) * 0.2 + 0.3
    large_data_gpu = large_data_cpu

    # GPU 相关配置
    threadsperblock = (16, 16)
    blockspergrid_x_dummy = (dummy_data.shape[0] + threadsperblock[0] -
                             1) // threadsperblock[0]
    blockspergrid_y_dummy = (dummy_data.shape[1] + threadsperblock[1] -
                             1) // threadsperblock[1]
    blockspergrid_dummy = (blockspergrid_x_dummy, blockspergrid_y_dummy)

    blockspergrid_x_large = (data_size[0] + threadsperblock[0] -
                             1) // threadsperblock[0]
    blockspergrid_y_large = (data_size[1] + threadsperblock[1] -
                             1) // threadsperblock[1]
    blockspergrid_large = (blockspergrid_x_large, blockspergrid_y_large)

    # 运行两次测试
    for run_idx in range(1, 3):
        print(f"\n{'='*10} 第 {run_idx} 次运行 {'='*10}")
        print("--- Numba 冷启动 (编译) 时间测试 ---")

        # --- CPU Numba (njit + prange) 冷启动测试 ---
        print("\n#### CPU Numba (njit + prange) ####")
        print("首次调用 (冷启动/编译时间):")
        _, cold_start_time_cpu = calculate_time_wrapper(complex_cpu_calc)(
            dummy_data)
        print(f"  冷启动/编译时间: {cold_start_time_cpu:.6f} 秒")

        # --- GPU Numba (CUDA Kernel) 冷启动测试 ---
        print("\n#### GPU Numba (CUDA Kernel) ####")
        if not cuda.is_available():
            print("CUDA 不可用，跳过 GPU 测试。")
            if run_idx == 1:
                break
            continue

        print("首次调用 (冷启动/编译时间):")
        d_dummy_in = cuda.to_device(dummy_data)
        d_dummy_out = cuda.device_array_like(d_dummy_in)
        _, cold_start_time_gpu = calculate_time_wrapper(
            lambda: complex_gpu_kernel[blockspergrid_dummy, threadsperblock]
            (d_dummy_in, d_dummy_out))()
        cuda.synchronize()
        print(f"  冷启动/编译时间: {cold_start_time_gpu:.6f} 秒")

        print("\n--- 实际计算时间测试 (使用更大规模数据) ---")

        # --- CPU Numba (njit + prange) 实际计算时间 ---
        print("\n#### CPU Numba (njit + prange) ####")
        _, calc_time_cpu = calculate_time_wrapper(complex_cpu_calc)(
            large_data_cpu)
        print(
            f"  实际计算时间 ({data_size[0]}x{data_size[1]} 数据): {calc_time_cpu:.6f} 秒"
        )

        # --- GPU Numba (CUDA Kernel) 实际计算时间 ---
        if cuda.is_available():
            print("\n#### GPU Numba (CUDA Kernel) ####")
            d_large_in = cuda.to_device(large_data_gpu)
            d_large_out = cuda.device_array_like(d_large_in)

            _, calc_time_gpu_transfer = calculate_time_wrapper(
                lambda: complex_gpu_kernel[blockspergrid_large, threadsperblock
                                           ](d_large_in, d_large_out))()
            cuda.synchronize()
            print(
                f"  实际计算时间 (含传输, {data_size[0]}x{data_size[1]} 数据): {calc_time_gpu_transfer:.6f} 秒"
            )

            start_gpu_event = cuda.event()
            end_gpu_event = cuda.event()

            start_gpu_event.record()
            complex_gpu_kernel[blockspergrid_large,
                               threadsperblock](d_large_in, d_large_out)
            end_gpu_event.record()
            end_gpu_event.synchronize()
            pure_kernel_time_gpu = cuda.event_elapsed_time(
                start_gpu_event, end_gpu_event) / 1000.0
            print(
                f"  GPU 纯内核执行时间 ({data_size[0]}x{data_size[1]} 数据): {pure_kernel_time_gpu:.6f} 秒"
            )


if __name__ == "__main__":
    run_tests()
