import numpy as np
import time

# 定义数组的维度
CONF_COUNT = 200
ROWS = 40000
OUTPUT_DIM = 20
DTYPE = np.float64  # 使用 float64，因为它在你的代码中很常见

# 计算总元素数量和总内存大小
total_elements = CONF_COUNT * ROWS * OUTPUT_DIM
memory_size_gb = total_elements * DTYPE().itemsize / (1024**3)

print(f"测试数组维度: ({CONF_COUNT}, {ROWS}, {OUTPUT_DIM})")
print(f"总元素数量: {total_elements:,}")
print(f"单个元素大小: {DTYPE().itemsize} 字节")
print(f"总内存占用: {memory_size_gb:.2f} GB\n")

# --- 场景 1: 每次都创建新的 NumPy 数组 (np.full) ---
print("--- 场景 1: 每次都创建新的 NumPy 数组 (np.full) ---")
num_runs = 5  # 重复运行多次，取平均值
creation_times = []

for i in range(num_runs):
    start_time = time.perf_counter()  # 使用 perf_counter 更精确
    new_array = np.full((CONF_COUNT, ROWS, OUTPUT_DIM), np.nan, dtype=DTYPE)
    end_time = time.perf_counter()
    duration = end_time - start_time
    creation_times.append(duration)
    print(f"第 {i+1} 次创建耗时: {duration:.6f} 秒")

print(f"平均创建耗时 (np.full): {np.mean(creation_times):.6f} 秒\n")

# --- 场景 2: 利用旧数组，原地修改 (.fill()) ---
print("--- 场景 2: 利用旧数组，原地修改 (.fill()) ---")
# 首先创建一个初始数组
existing_array = np.full((CONF_COUNT, ROWS, OUTPUT_DIM), 0.0, dtype=DTYPE)
print(
    f"已创建初始数组，准备进行原地修改。其内存地址: {existing_array.__array_interface__['data'][0]}\n"
)

fill_times = []
for i in range(num_runs):
    start_time = time.perf_counter()
    existing_array.fill(np.nan)  # 填充为 NaN
    end_time = time.perf_counter()
    duration = end_time - start_time
    fill_times.append(duration)
    print(f"第 {i+1} 次原地填充耗时: {duration:.6f} 秒")

print(f"平均原地填充耗时 (.fill()): {np.mean(fill_times):.6f} 秒")
print(f"注意观察内存地址，确保是同一个数组被修改: {existing_array.__array_interface__['data'][0]}")
