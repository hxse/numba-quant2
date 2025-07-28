from numba import cuda
import numpy as np


# --- 1. GPU设备函数：实际的计算和写入多列结果 ---
@cuda.jit(device=True)
def calculate_and_write_multiple_columns_device(
        output_columns_slice,  # 现在接收一个二维切片，包含所有目标列
        input_val,  # 输入值
        kline_idx  # 当前K线索引
):
    """
    这个设备函数现在接收一个**二维切片**，代表了多列结果。
    它负责将计算结果写入这个二维切片的不同列中。
    """
    # 获取切片中第一列的视图 (对应原始统一数组的 start_col_idx 列)
    # output_columns_slice 的第一维是行，第二维是列
    col0_in_slice = output_columns_slice[:, 0]

    # 获取切片中第二列的视图 (对应原始统一数组的 start_col_idx + 1 列)
    col1_in_slice = output_columns_slice[:, 1]

    # 模拟计算并写入到切片中的第一列
    col0_in_slice[kline_idx] = input_val * 2.0

    # 模拟计算并写入到切片中的第二列
    col1_in_slice[kline_idx] = input_val + 10.0


# --- 2. GPU主 Kernel：在其中进行多列切片并作为单个参数传递 ---
@cuda.jit
def main_kernel(input_data, unified_output_array, start_col_idx, stop_col_idx):
    """
    主 CUDA Kernel，每个线程处理一个K线数据点。
    它负责在调用 calculate_and_write_multiple_columns_device 之前进行多列切片。
    """
    kline_idx = cuda.grid(1)

    if kline_idx < input_data.shape[0]:
        data_val = input_data[kline_idx]

        # --- 核心部分：在这里进行多列切片，并将其作为一个二维视图 ---
        # unified_output_array[:, start_col_idx:stop_col_idx]
        # 得到的是一个二维 DeviceNDArray 视图，包含从 start_col_idx 到 stop_col_idx-1 的所有列。
        # 注意：Python 切片是左闭右开的。
        target_columns_slice = unified_output_array[:,
                                                    start_col_idx:stop_col_idx]

        # --- 通过单个参数传递二维切片调用设备函数 ---
        calculate_and_write_multiple_columns_device(
            target_columns_slice,  # 传递包含所有目标列的二维切片
            data_val,
            kline_idx)


# --- 3. CPU 端：准备数据和执行 ---
if __name__ == '__main__':
    N_KLINE = 5  # K线数量
    NUM_TOTAL_COLS = 10  # 统一结果数组的总列数

    # 模拟输入数据
    input_data_cpu = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    d_input_data = cuda.to_device(input_data_cpu)

    # 预定义统一的二维结果数组
    unified_results_cpu = np.full((N_KLINE, NUM_TOTAL_COLS),
                                  np.nan,
                                  dtype=np.float64)
    d_unified_results = cuda.to_device(unified_results_cpu)

    # 指定我们希望写入的起始和结束列索引。
    # 例如，希望写入统一数组的第3列和第4列 (索引为2和3)。
    # Python 切片是左闭右开，所以 start=2, stop=4 对应列索引 2, 3。
    start_column_for_output = 2
    stop_column_for_output = 4  # Numba 切片 [start:stop) 意味着包含 start, 不包含 stop

    # 设置 CUDA 核函数的网格和块大小
    threads_per_block = 5
    blocks_per_grid = (N_KLINE + threads_per_block - 1) // threads_per_block

    print("--- 启动 CUDA Kernel ---")
    main_kernel[blocks_per_grid,
                threads_per_block](d_input_data, d_unified_results,
                                   start_column_for_output,
                                   stop_column_for_output)
    cuda.synchronize()

    # 将结果拷贝回CPU
    final_results_host = d_unified_results.copy_to_host()

    print("\n--- 原始输入数据 ---")
    print(input_data_cpu)

    print(
        f"\n--- 统一结果数组 (包含计算结果，写入列 {start_column_for_output} 到 {stop_column_for_output-1}) ---"
    )
    print(final_results_host)

    # 验证结果
    print(
        f"\n--- 预期结果 (对应原始统一数组的列 {start_column_for_output} 和 {start_column_for_output + 1}) ---"
    )
    expected_col0_in_slice = input_data_cpu * 2.0
    expected_col1_in_slice = input_data_cpu + 10.0
    print(
        f"预期切片内第0列 (原始列 {start_column_for_output}): {expected_col0_in_slice}")
    print(
        f"预期切片内第1列 (原始列 {start_column_for_output + 1}): {expected_col1_in_slice}"
    )

    print("\n--- 从统一结果数组中提取和验证 ---")
    retrieved_col0 = final_results_host[:, start_column_for_output]
    retrieved_col1 = final_results_host[:, start_column_for_output + 1]
    print(f"实际提取列 {start_column_for_output}: {retrieved_col0}")
    print(f"实际提取列 {start_column_for_output + 1}: {retrieved_col1}")

    # 检查是否与预期一致
    assert np.allclose(retrieved_col0, expected_col0_in_slice)
    assert np.allclose(retrieved_col1, expected_col1_in_slice)
    print("\n验证成功：计算结果与预期一致！")
