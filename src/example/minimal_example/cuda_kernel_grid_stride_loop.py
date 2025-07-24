from numba import cuda
import numba as nb
import numpy as np


# 1. 修改后的 CUDA 核函数，采用网格步进循环
@nb.cuda.jit
def simple_test_kernel_with_stride_loop(output_array, total_tasks):
    # 获取当前线程的全局起始索引
    start_idx = nb.cuda.grid(1)
    # 获取整个网格中启动的线程总数，作为步长
    stride = nb.cuda.gridsize(1)

    # 使用 for 循环，让当前线程处理其负责的所有任务
    # 循环从 start_idx 开始，每次跳跃 stride，直到达到 total_tasks
    for idx in range(start_idx, total_tasks, stride):
        # 执行任务：将任务的索引乘以10加1，并存入结果数组
        output_array[idx] = idx * 10 + 1


# 2. 主机函数，与之前基本相同，只是调用修改后的核函数
def run_minimal_test_with_stride_loop():
    total_tasks = 10  # 我们总共有 10 个任务需要处理

    # 启动配置：只启动 4 个线程（和之前一样，刻意让线程数少于任务数）
    threads_per_block = 4  # 每个线程块 4 个线程
    blocks_per_grid = 1  # 只启动 1 个线程块

    # 计算实际启动的线程总数
    total_threads_launched = threads_per_block * blocks_per_grid

    # 准备结果数组，初始值设为 -1
    results_host = np.full(total_tasks, -1, dtype=np.int32)

    print(f"--- 测试配置 (带步进循环) ---")
    print(f"总任务数 (total_tasks): {total_tasks}")
    print(f"每个线程块线程数 (threads_per_block): {threads_per_block}")
    print(f"线程块数量 (blocks_per_grid): {blocks_per_grid}")
    print(f"实际启动的 GPU 线程总数: {total_threads_launched}")
    print(f"初始结果数组: {results_host}")

    # 将结果数组从主机内存复制到 GPU 设备内存
    d_results = nb.cuda.to_device(results_host)

    # 启动 GPU 核函数 (注意这里调用的是带有步进循环的新核函数)
    simple_test_kernel_with_stride_loop[blocks_per_grid,
                                        threads_per_block](d_results,
                                                           total_tasks)

    # 强制 GPU 同步，确保所有计算完成
    nb.cuda.synchronize()

    # 将结果从 GPU 设备内存复制回主机内存
    results_final = d_results.copy_to_host()

    print(f"\n--- 执行结果 (带步进循环) ---")
    print(f"核函数执行后的结果数组: {results_final}")

    # 验证结果
    print("\n--- 验证 ---")
    expected_results = np.arange(total_tasks) * 10 + 1
    if np.array_equal(results_final, expected_results):
        print(f"✅ 验证通过：所有 {total_tasks} 个任务都被正确处理了！")
    else:
        print("❌ 验证失败：结果不符合预期。")


# 运行测试
if __name__ == "__main__":
    run_minimal_test_with_stride_loop()
