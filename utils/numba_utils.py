from numba import jit, njit, cuda
import numpy as np

_compiled_functions_cache = {}


def numba_wrapper(
    mode: str,
    signature: tuple | None = None,  # 签名现在可以为 None，因为 normal 模式不需要
    cache_enabled: bool = False,  # 默认改为 False，按需开启
    parallel: bool = False,
    set_inline_to_always: bool = False,
    max_registers: int | None = None,
    use_global_cache: bool = True,
):
    """
    根据指定的模式返回一个 Numba 装饰器工厂。
    增加了全局缓存机制，避免重复编译/获取已缓存函数时的开销。

    Args:
        mode (str): Numba 模式，可以是 'jit', 'njit', 或 'cuda'。
        signature (tuple): Numba 装饰器的签名参数，例如 (float64[:], float64[:])。
        cache_enabled (bool): Numba 自身的缓存机制（文件系统缓存）是否启用。
        parallel (bool): 是否启用并行化（对 njit 和 jit 有效）。
        max_registers (int | None): CUDA 模式下的最大寄存器数。
        use_global_cache (bool): 是否启用本包装函数内部的全局字典缓存。
                                  如果为 True，函数将尝试从内存字典中获取已编译函数。

    Returns:
        callable: 一个内部装饰器函数，用于装饰目标函数。

    Raises:
        ValueError: 如果模式无效。
        ImportError: 如果尝试使用 CUDA 模式但 CUDA 不可用。
    """

    def decorator(func):
        # 构建缓存键的基础元素
        key_elements = [
            func.__qualname__,
            mode,
            signature,
            cache_enabled,
            parallel,
            set_inline_to_always,
        ]

        # max_registers 只在 CUDA 模式下影响编译结果，所以只在这种情况下加入到键中
        if mode == "cuda":
            key_elements.append(max_registers)

        cache_key = tuple(key_elements)

        # 检查全局缓存
        if use_global_cache and cache_key in _compiled_functions_cache:
            # print(f"--- 从全局缓存获取函数: {func.__qualname__} ---") # 调试信息
            return _compiled_functions_cache[cache_key]

        # 动态构建 Numba 装饰器的参数字典
        decorator_kwargs = {"parallel": parallel, "cache": cache_enabled}

        if set_inline_to_always:
            decorator_kwargs["inline"] = "always"  # 'always' or 'never'

        # 如果不在全局缓存中，则执行 Numba 编译
        if mode == "normal":
            decorator_kwargs["nopython"] = False
            compiled_func = jit(signature, **decorator_kwargs)(func)
        elif mode == "njit":
            compiled_func = njit(signature, **decorator_kwargs)(func)
        elif mode == "cuda":
            decorator_kwargs["device"] = not parallel
            if max_registers is not None:
                decorator_kwargs["max_registers"] = max_registers
            decorator_kwargs.pop("parallel", None)  # 移除不适用的参数
            compiled_func = cuda.jit(signature, **decorator_kwargs)(func)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # 如果启用了全局缓存，则将编译好的函数存入缓存
        if use_global_cache:
            _compiled_functions_cache[cache_key] = compiled_func

        return compiled_func

    return decorator
