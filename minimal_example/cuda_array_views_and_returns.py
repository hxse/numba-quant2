import numba as nb
from numba import cuda
import numpy as np


# --- 1. 设备函数：尝试直接返回索引的一行 (一维数组视图) - 会报错 ---
@cuda.jit(device=True)
def get_row_direct_indexed_ERROR(matrix, row_idx):
    """
    尝试直接返回二维数组的某一行（通过直接索引），这将导致 TypingError。
    """
    return matrix[row_idx]  # 预期报错


# --- 2. 设备函数：尝试直接返回切片的一行 (一维数组视图) - 会报错 ---
@cuda.jit(device=True)
def get_row_direct_sliced_ERROR(matrix, row_idx):
    """
    尝试直接返回二维数组的某一行（通过切片），这将导致 TypingError。
    """
    return matrix[row_idx, :]  # 预期报错


# --- 3. 设备函数：返回元组包裹的直接索引的一行 (可以工作) ---
@cuda.jit(device=True)
def get_row_tuple_indexed_OK(matrix, row_idx):
    """
    返回元组包裹的二维数组某一行（通过直接索引），这是允许的。
    """
    return (matrix[row_idx], )


# --- 4. 设备函数：返回元组包裹的切片的一行 (可以工作) ---
@cuda.jit(device=True)
def get_row_tuple_sliced_OK(matrix, row_idx):
    """
    返回元组包裹的二维数组某一行（通过切片），这是允许的。
    """
    return (matrix[row_idx, :], )


# --- 通用处理函数：对获取到的行求和 ---
@cuda.jit(device=True)
def process_row_data(row_data_view):
    """
    接收一个一维数组视图，并对其所有元素求和。
    """
    sum_val = 0.0
    for i in range(row_data_view.shape[0]):
        sum_val += row_data_view[i]
    return sum_val


# --- 内核函数 1：测试直接索引返回 (预期报错) ---
@cuda.jit
def kernel_test_direct_indexed_error(input_matrix, output_results):
    idx = cuda.grid(1)
    if idx < input_matrix.shape[0]:
        row_view = get_row_direct_indexed_ERROR(input_matrix, idx)  # 编译时会报错
        output_results[idx] = process_row_data(row_view)


# --- 内核函数 2：测试直接切片返回 (预期报错) ---
@cuda.jit
def kernel_test_direct_sliced_error(input_matrix, output_results):
    idx = cuda.grid(1)
    if idx < input_matrix.shape[0]:
        row_view = get_row_direct_sliced_ERROR(input_matrix, idx)  # 编译时会报错
        output_results[idx] = process_row_data(row_view)


# --- 内核函数 3：测试元组包裹直接索引返回 (预期成功) ---
@cuda.jit
def kernel_test_tuple_indexed_ok(input_matrix, output_results):
    idx = cuda.grid(1)
    if idx < input_matrix.shape[0]:
        row_tuple = get_row_tuple_indexed_OK(input_matrix, idx)
        row_view = row_tuple[0]  # 从元组中解包出视图
        output_results[idx] = process_row_data(row_view)


# --- 内核函数 4：测试元组包裹切片返回 (预期成功) ---
@cuda.jit
def kernel_test_tuple_sliced_ok(input_matrix, output_results):
    idx = cuda.grid(1)
    if idx < input_matrix.shape[0]:
        row_tuple = get_row_tuple_sliced_OK(input_matrix, idx)
        row_view = row_tuple[0]  # 从元组中解包出视图
        output_results[idx] = process_row_data(row_view)


# --- 主机端代码执行 ---
if __name__ == "__main__":
    rows = 10
    cols = 5
    cpu_input_matrix = np.random.rand(rows, cols).astype(np.float32)
    gpu_input_matrix = cuda.to_device(cpu_input_matrix)

    threads_per_block = 256
    blocks_per_grid = (rows + (threads_per_block - 1)) // threads_per_block

    print(f"--- CPU 输入矩阵 ---\n{cpu_input_matrix}\n")

    # --- 运行测试 1：直接返回索引的行 (预期报错) ---
    print("--- 测试 1: 直接返回索引的行 (ERROR) ---")
    gpu_output_error1 = cuda.device_array(rows, dtype=np.float32)
    try:
        kernel_test_direct_indexed_error[blocks_per_grid,
                                         threads_per_block](gpu_input_matrix,
                                                            gpu_output_error1)
        cuda.synchronize()
        print("状态: 意外成功（请检查 Numba 版本或环境）\n")
    except nb.core.errors.TypingError:
        print("状态: 成功捕获到预期的 TypingError。\n")
    except Exception as e:
        print(f"状态: 捕获到其他错误: {type(e).__name__}\n")

    # --- 运行测试 2：直接返回切片的行 (ERROR) ---
    print("--- 测试 2: 直接返回切片的行 (ERROR) ---")
    gpu_output_error2 = cuda.device_array(rows, dtype=np.float32)
    try:
        kernel_test_direct_sliced_error[blocks_per_grid,
                                        threads_per_block](gpu_input_matrix,
                                                           gpu_output_error2)
        cuda.synchronize()
        print("状态: 意外成功（请检查 Numba 版本或环境）\n")
    except nb.core.errors.TypingError:
        print("状态: 成功捕获到预期的 TypingError。\n")
    except Exception as e:
        print(f"状态: 捕获到其他错误: {type(e).__name__}\n")

    # --- 运行测试 3：返回元组包裹的直接索引的行 (OK) ---
    print("--- 测试 3: 返回元组包裹的直接索引的行 (OK) ---")
    gpu_output_ok3 = cuda.device_array(rows, dtype=np.float32)
    try:
        kernel_test_tuple_indexed_ok[blocks_per_grid,
                                     threads_per_block](gpu_input_matrix,
                                                        gpu_output_ok3)
        cuda.synchronize()
        cpu_output_ok3 = gpu_output_ok3.copy_to_host()
        expected_results = np.sum(cpu_input_matrix, axis=1)
        if np.allclose(cpu_output_ok3, expected_results):
            print("状态: 成功并结果一致。\n")
        else:
            print("状态: 成功但结果不一致。\n")
    except Exception as e:
        print(f"状态: 运行出错: {type(e).__name__}\n")

    # --- 运行测试 4：返回元组包裹的切片的行 (OK) ---
    print("--- 测试 4: 返回元组包裹的切片的行 (OK) ---")
    gpu_output_ok4 = cuda.device_array(rows, dtype=np.float32)
    try:
        kernel_test_tuple_sliced_ok[blocks_per_grid,
                                    threads_per_block](gpu_input_matrix,
                                                       gpu_output_ok4)
        cuda.synchronize()
        cpu_output_ok4 = gpu_output_ok4.copy_to_host()
        expected_results = np.sum(cpu_input_matrix, axis=1)
        if np.allclose(cpu_output_ok4, expected_results):
            print("状态: 成功并结果一致。\n")
        else:
            print("状态: 成功但结果不一致。\n")
    except Exception as e:
        print(f"状态: 运行出错: {type(e).__name__}\n")
