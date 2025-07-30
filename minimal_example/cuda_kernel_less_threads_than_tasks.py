from numba import cuda
import numba as nb
import numpy as np


# 1. 定义一个简单的 CUDA 核函数
# 这个核函数会尝试将任务的索引乘以10加1，并存入结果数组
# 它严格按照传入的 idx 来处理，没有内部循环
@nb.cuda.jit
def simple_test_kernel(output_array, total_tasks):
    # 获取当前线程在整个网格中的唯一ID (0, 1, 2, 3...)
    idx = nb.cuda.grid(1)

    # 只有当线程ID小于总任务数时才执行任务
    if idx < total_tasks:
        output_array[idx] = idx * 10 + 1  # 执行一个简单的计算


# 2. 定义一个主机函数来启动 GPU 核函数并验证结果
def run_minimal_test():
    total_tasks = 10  # 我们总共有 10 个任务需要处理

    # 启动配置：只启动 4 个线程
    threads_per_block = 4  # 每个线程块 4 个线程
    blocks_per_grid = 1  # 只启动 1 个线程块

    # 计算实际启动的线程总数
    total_threads_launched = threads_per_block * blocks_per_grid

    # 准备结果数组，初始值设为 -1，方便看出哪些任务未被处理
    # 数组大小为总任务数
    results_host = np.full(total_tasks, -1, dtype=np.int32)

    print(f"--- 测试配置 ---")
    print(f"总任务数 (total_tasks): {total_tasks}")
    print(f"每个线程块线程数 (threads_per_block): {threads_per_block}")
    print(f"线程块数量 (blocks_per_grid): {blocks_per_grid}")
    print(f"实际启动的 GPU 线程总数: {total_threads_launched}")
    print(f"初始结果数组: {results_host}")

    # 将结果数组从主机内存复制到 GPU 设备内存
    d_results = nb.cuda.to_device(results_host)

    # 启动 GPU 核函数
    # [blocks_per_grid, threads_per_block] 定义了启动配置
    simple_test_kernel[blocks_per_grid, threads_per_block](d_results, total_tasks)

    # 强制 GPU 同步，确保所有计算完成
    nb.cuda.synchronize()

    # 将结果从 GPU 设备内存复制回主机内存
    results_final = d_results.copy_to_host()

    print(f"\n--- 执行结果 ---")
    print(f"核函数执行后的结果数组: {results_final}")

    # 验证结果
    print("\n--- 验证 ---")
    if np.array_equal(
        results_final[:total_threads_launched],
        np.arange(total_threads_launched) * 10 + 1,
    ) and np.all(results_final[total_threads_launched:] == -1):
        print(f"✅ 验证通过：只有前 {total_threads_launched} 个任务被处理了。")
        print(
            f"   任务 {total_threads_launched} 到 {total_tasks - 1} (即任务 4 到 9) 未被处理，值为 -1。"
        )
    else:
        print("❌ 验证失败：结果不符合预期。")


# 运行测试
if __name__ == "__main__":
    run_minimal_test()
