import numpy as np
import numba as nb
import time
from functools import wraps

# --- 1. 全局缓存字典 ---
# 此字典用于存储 Numba 编译后的函数。它是本示例中用于消除工厂函数重复开销的关键。
_compiled_functions_cache = {}


# --- 2. 测量函数耗时的装饰器 ---
def time_wrapper(func):
    """
    一个装饰器，用于测量其所装饰的**工厂函数**的执行时间。
    这帮助我们量化获取 Numba 编译函数本身的开销。
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        # 打印工厂函数的运行时间，明确指出这是“工厂函数调用”的开销
        print(f"  {func.__name__} 运行时间 (工厂函数调用): {run_time:.6f} 秒")
        return result

    return wrapper


# --- 3. Numba 装饰器工厂，包含全局缓存逻辑 ---
def numba_wrapper(
        mode: str,
        signature: tuple,
        cache_enabled: bool,
        parallel: bool = False,
        max_registers: int | None = None,  # 保持参数签名，但在 'njit' 模式下不使用
        use_global_cache: bool = True):
    """
    根据指定的 Numba 模式和配置返回一个装饰器。
    核心目的是集成**自定义的全局缓存机制**，以避免 Numba 编译函数时的重复开销。
    此示例仅聚焦于 'njit' 模式，这是 Numba 推荐的高性能模式。
    """

    def decorator(func):
        # 构建唯一的缓存键：确保每个Numba函数及其特定编译配置（签名、并行等）
        # 都有一个独一无二的标识。这是全局缓存正确工作的基石。
        key_elements = [
            func.__qualname__, mode, signature, parallel, cache_enabled
        ]
        # max_registers 仅在 CUDA 模式下相关，此处忽略以简化缓存键
        cache_key = tuple(key_elements)

        # 核心逻辑：检查全局缓存
        # 如果启用全局缓存且该函数已存在于缓存中，则直接返回缓存的函数。
        # 这是消除工厂函数重复开销的关键步骤。
        if use_global_cache and cache_key in _compiled_functions_cache:
            return _compiled_functions_cache[cache_key]

        # 如果不在全局缓存中，则调用 Numba 进行实际的编译（或从 Numba 自身的文件缓存加载）
        if mode == 'njit':
            compiled_func = nb.njit(signature,
                                    parallel=parallel,
                                    cache=cache_enabled)(func)
        else:
            raise ValueError("此示例仅支持 'njit' 模式。")

        # 将 Numba 编译后的函数存储到我们的全局字典缓存中，以便后续快速检索
        if use_global_cache:
            _compiled_functions_cache[cache_key] = compiled_func

        return compiled_func

    return decorator


# --- 4. 经典的 Numba 问题函数：迭代矩阵计算 ---
# 注意：已移除 @nb.njit 装饰器，该装饰器现在由 numba_wrapper 动态应用
def _classic_numba_problem(matrix, iterations):
    """
    一个经典的 Numba 优化问题：对二维数组进行迭代计算。
    此函数的**内部结构被设计得较为复杂**，以增加 Numba 编译器的分析负担。
    其目的是为了**更明显地体现工厂函数在没有自定义全局缓存时的重复开销**，
    因为 Numba 每次“加载”或“确认”这样一个复杂函数时，本身就需要时间。
    """
    rows, cols = matrix.shape
    result_matrix = np.copy(matrix)  # 创建副本，防止修改原始输入

    # 循环模拟迭代计算过程
    for _ in range(iterations):
        for i in range(rows):
            for j in range(cols):
                # 简单迭代计算，涉及相邻元素，增加数据依赖性
                val_current = result_matrix[i, j]
                val_next_col = result_matrix[i, (j + 1) % cols]  # 循环访问下一列
                val_next_row = result_matrix[(i + 1) % rows, j]  # 循环访问下一行

                # 增加浮点运算和条件分支，使函数体更复杂，增大 Numba 的编译分析量
                if val_current > 0.5:
                    result_matrix[i, j] = (val_current * 0.99 + val_next_col *
                                           0.005 + val_next_row * 0.005) * 1.01
                else:
                    result_matrix[i, j] = (val_current * 0.98 + val_next_col *
                                           0.01 + val_next_row * 0.01) * 0.99
    # 返回稍微复杂化的结果，确保 Numba 处理输出
    return result_matrix * 1.05 + iterations / 1000.0


# --- 5. 工厂函数：负责返回编译后的 Numba 函数 ---
@time_wrapper
def get_classic_numba_func(cache_enabled: bool = True,
                           use_global_cache: bool = True):
    """
    这是一个**工厂函数**。它不直接执行计算，而是负责调用 `numba_wrapper`
    来获取（或编译）并返回一个 Numba 编译过的 `_classic_numba_problem` 函数。
    本示例的重点是**测量这个工厂函数本身的调用时间**，以对比全局缓存的效果。
    """
    # 定义 Numba 函数的输入参数签名
    signature = (nb.float64[:, :], nb.int64)  # 接受一个二维浮点数组和一个整数

    # 调用 numba_wrapper 获取编译后的函数
    compiled_func = numba_wrapper(
        mode="njit",
        signature=signature,
        cache_enabled=cache_enabled,  # 控制 Numba 自身的文件缓存
        use_global_cache=use_global_cache  # 控制我们自定义的全局字典缓存
    )(_classic_numba_problem)
    return compiled_func


# --- 6. 主测试逻辑 ---
if __name__ == "__main__":
    # 定义输入数据的大小和内部迭代次数，这些参数会影响 _classic_numba_problem 的计算量，
    # 进而影响 Numba 编译器的分析时间和所生成的代码大小，从而影响工厂函数的开销。
    matrix_rows = 1000
    matrix_cols = 1000
    num_iterations = 10

    # 生成模拟输入数据
    print(f"生成 {matrix_rows}x{matrix_cols} 的模拟矩阵数据...")
    input_matrix = np.random.rand(matrix_rows, matrix_cols).astype(np.float64)
    print("数据生成完毕。")

    print("\n--- 场景 1: 使用全局缓存 (use_global_cache=True) ---")
    # 在每个场景开始前清除自定义的全局缓存，确保测试的独立性。
    _compiled_functions_cache.clear()

    print("\n--- 第一次调用工厂函数 (Numba 编译/文件缓存加载 + 全局缓存写入) ---")
    # 首次调用工厂函数：
    # Numba 会进行实际的编译工作（如果其文件缓存中没有），并生成机器码。
    # 我们的全局缓存会捕获这个编译结果。
    factory_func_1_on = get_classic_numba_func(cache_enabled=True,
                                               use_global_cache=True)
    # 立即执行一次 Numba 编译后的函数，这会触发 Numba 的惰性 JIT 编译，
    # 确保编译过程完成，但我们不测量其执行时间，只关注工厂函数时间。
    factory_func_1_on(input_matrix, num_iterations)

    print("\n--- 第二次调用工厂函数 (从全局缓存快速获取) ---")
    # 再次调用工厂函数：
    # 因为全局缓存已启用且已命中，这次调用会直接从内存中的 _compiled_functions_cache 获取，
    # 因此预期耗时会非常短，接近零。
    factory_func_2_on = get_classic_numba_func(cache_enabled=True,
                                               use_global_cache=True)
    # 再次执行编译后的函数，仅为保持测试流程的完整性。
    factory_func_2_on(input_matrix, num_iterations)

    print("\n\n--- 场景 2: 关闭全局缓存 (use_global_cache=False) ---")
    # 彻底清除全局缓存，模拟一个全新的程序启动状态，此时程序只能依赖 Numba 自身的文件缓存。
    _compiled_functions_cache.clear()

    # 重新生成模拟数据，以保证测试环境的独立性，避免数据副作用。
    print("重新生成模拟数据 (为场景 2)...")
    input_matrix_off = np.random.rand(matrix_rows,
                                      matrix_cols).astype(np.float64)
    print("数据生成完毕。")

    print("\n--- 第一次调用工厂函数 (Numba 编译/文件缓存加载) ---")
    # 首次调用工厂函数：
    # Numba 会进行编译或从其文件缓存加载。由于我们禁用了自定义全局缓存，
    # 即使 Numba 内部有缓存，每次工厂函数调用仍需通过 Numba 的内部机制来获取函数。
    factory_func_1_off = get_classic_numba_func(cache_enabled=True,
                                                use_global_cache=False)
    factory_func_1_off(input_matrix_off, num_iterations)

    print("\n--- 第二次调用工厂函数 (仅依赖 Numba 自身文件缓存，应有开销) ---")
    # 再次调用工厂函数：
    # 这是本示例中**最关键的对比点**。
    # 即使 Numba 自身文件缓存可能已生效，但由于我们没有自定义的全局内存缓存，
    # 每次调用工厂函数时，Numba 仍然需要执行一些内部查找、加载或验证操作。
    # 因此，预期会观察到**显著的重复开销**，明显高于使用全局缓存时的第二次调用。
    factory_func_2_off = get_classic_numba_func(cache_enabled=True,
                                                use_global_cache=False)
    factory_func_2_off(input_matrix_off, num_iterations)
