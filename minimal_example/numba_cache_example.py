# numba_cache_example.py

import os
import shutil
import time
from numba import jit, njit, cuda
import numpy as np
from numba import float64, int64  # 导入特定类型用于签名
import argparse  # 导入 argparse 模块


# --- 你的 numba_wrapper 工具函数（原封不动地复制过来） ---
_compiled_functions_cache = {}


def numba_wrapper(
    mode: str,
    signature: tuple | None = None,
    cache_enabled: bool = False,
    parallel: bool = False,
    set_inline_to_always: bool = False,
    max_registers: int | None = None,
    use_global_cache: bool = True,
):
    def decorator(func):
        key_elements = [
            func.__qualname__,
            mode,
            signature,
            cache_enabled,
            parallel,
            set_inline_to_always,
        ]
        if mode == "cuda":
            key_elements.append(max_registers)
        cache_key = tuple(key_elements)

        if use_global_cache and cache_key in _compiled_functions_cache:
            print(
                f"[WRAPPER_CACHE] {func.__qualname__} from GLOBAL MEMORY CACHE (ID: {id(_compiled_functions_cache[cache_key])})"
            )
            return _compiled_functions_cache[cache_key]

        decorator_kwargs = {"parallel": parallel, "cache": cache_enabled}

        if set_inline_to_always:
            decorator_kwargs["inline"] = "always"

        if mode == "normal":
            decorator_kwargs["nopython"] = False
            compiled_func = jit(signature, **decorator_kwargs)(func)
        elif mode == "njit":
            compiled_func = njit(signature, **decorator_kwargs)(func)
        elif mode == "cuda":
            decorator_kwargs["device"] = not parallel
            if max_registers is not None:
                decorator_kwargs["max_registers"] = max_registers
            decorator_kwargs.pop("parallel", None)
            compiled_func = cuda.jit(signature, **decorator_kwargs)(func)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if use_global_cache:
            _compiled_functions_cache[cache_key] = compiled_func

        print(
            f"[WRAPPER_COMPILATION] {func.__qualname__} COMPILED (ID: {id(compiled_func)})"
        )
        return compiled_func

    return decorator


# --- 辅助函数：清除 Numba 缓存 ---
def clear_numba_cache_dirs():
    """递归清除当前脚本所在目录及其所有子目录下的 __pycache__ 目录。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for root, dirs, files in os.walk(script_dir):
        if "__pycache__" in dirs:
            cache_path = os.path.join(root, "__pycache__")
            shutil.rmtree(cache_path)
            print(f"清除了 {cache_path} 目录下的 Numba 缓存。")


# --- 内部 Numba 函数的工厂函数 ---
def inner_func_factory(mode, cache_enabled):
    def _inner_calc(x):
        return x * 2 + 10

    signature = float64(float64)

    return numba_wrapper(mode, signature=signature, cache_enabled=cache_enabled)(
        _inner_calc
    )


# --- 外部 Numba 函数的工厂函数（复现问题） ---
def outer_func_factory_problematic(mode, cache_enabled):
    # 每次调用 outer_func_factory_problematic 都会重新创建 inner_njit_func 的新实例
    inner_njit_func = inner_func_factory(mode, cache_enabled)

    print(
        f"[OUTER_FACTORY] Problematic: inner_njit_func object ID: {id(inner_njit_func)}"
    )

    def _outer_calc_problematic(arr):
        total = 0.0
        for x in arr:
            total += inner_njit_func(x)  # 闭包捕获这个动态创建的实例
        return total

    signature = float64(float64[:])

    return numba_wrapper(
        mode, signature=signature, cache_enabled=cache_enabled, parallel=True
    )(_outer_calc_problematic)


# --- 解决问题的外部 Numba 函数的工厂函数 ---
# _cached_inner_njit_func 在模块级别只被创建一次
# 因此在所有对 outer_func_factory_solved 的调用中，它引用的是同一个 Python 对象实例
_cached_inner_njit_func = inner_func_factory("njit", cache_enabled=True)


def outer_func_factory_solved(mode, cache_enabled):
    print(
        f"[OUTER_FACTORY] Solved: _cached_inner_njit_func object ID: {id(_cached_inner_njit_func)}"
    )

    def _outer_calc_solved(arr):
        total = 0.0
        for x in arr:
            total += _cached_inner_njit_func(x)  # 闭包捕获稳定的实例
        return total

    signature = float64(float64[:])

    return numba_wrapper(
        mode, signature=signature, cache_enabled=cache_enabled, parallel=True
    )(_outer_calc_solved)


# --- 测试运行函数 ---
def run_scenario(factory_func, description, clear_cache_flag):
    print(f"\n--- 测试场景: {description} ---")

    # 根据命令行参数决定是否清除缓存
    if clear_cache_flag:
        clear_numba_cache_dirs()
    else:
        print("未清除 Numba 缓存。")

    start_time = time.time()
    compiled_func = factory_func("njit", cache_enabled=True)
    test_data = np.arange(1000, dtype=np.float64)
    result = compiled_func(test_data)
    end_time = time.time()
    print(f"结果: {result}, 运行时间: {end_time - start_time:.6f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Numba 缓存失效与解决示例。")
    parser.add_argument(
        "--clear-cache", "-c", action="store_true", help="在运行前清除 Numba 缓存。"
    )
    args = parser.parse_args()

    # 首先运行问题复现场景
    print("\n##### 场景一: Numba 缓存失效 (动态引用内部函数) #####")
    run_scenario(outer_func_factory_problematic, "问题复现", args.clear_cache)

    # 接着运行解决问题场景
    # 注意：为了让解决场景的缓存生效，第一次运行时必须在 clear_cache 模式下运行
    # 这样才能确保 _cached_inner_njit_func 被首次编译并缓存
    print("\n##### 场景二: Numba 缓存正常 (模块级别引用内部函数) #####")
    run_scenario(outer_func_factory_solved, "解决问题", args.clear_cache)

    print("\n--- 测试完成 ---")
    print("请手动执行多次以观察跨进程缓存效果。")
    print("第一次运行请使用 `python numba_cache_example.py -c` 来初始化缓存。")
    print("后续运行请使用 `python numba_cache_example.py` (不带 -c) 来观察缓存加载。")
