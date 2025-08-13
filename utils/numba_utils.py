from numba import jit, njit, cuda
import numpy as np


def nb_wrapper(
    mode: str,
    signature: tuple | None = None,  # 签名现在可以为 None，因为 normal 模式不需要
    cache_enabled: bool = False,  # 默认改为 False，按需开启
    parallel: bool = False,
    set_inline_to_always: bool = False,
    max_registers: int | None = None,
):
    # 动态构建 Numba 装饰器的参数字典
    decorator_kwargs = {"parallel": parallel, "cache": cache_enabled}

    if set_inline_to_always:
        decorator_kwargs["inline"] = "always"  # 'always' or 'never'

    # 如果不在全局缓存中，则执行 Numba 编译
    if mode == "normal":
        decorator_kwargs["nopython"] = False
        return jit(signature, **decorator_kwargs)
    elif mode == "njit":
        return njit(signature, **decorator_kwargs)
    elif mode == "cuda":
        decorator_kwargs["device"] = not parallel
        if max_registers is not None:
            decorator_kwargs["max_registers"] = max_registers
        decorator_kwargs.pop("parallel", None)  # 移除不适用的参数
        return cuda.jit(signature, **decorator_kwargs)
    else:
        raise ValueError(f"Invalid mode: {mode}")
