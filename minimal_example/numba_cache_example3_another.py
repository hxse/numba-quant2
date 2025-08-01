# numba_functions.py
from numba import jit, float64, float32
import json
import os

from numba_cache_example3_entry import numba_config_from_cli


nopython = numba_config_from_cli["nopython"]
cache = numba_config_from_cli["cache"]
enable64 = numba_config_from_cli["enable64"]
signal = numba_config_from_cli["signal"]

print("another", enable64)


# Numba JIT 编译函数，它们需要是模块级别的
@jit(signal, nopython=nopython, cache=cache)
def add2_numba(a, b):
    return a + b


@jit(signal, nopython=nopython, cache=cache)
def add_with_offset_numba(a, b):
    return add2_numba(a, b) + 1


# 提供一个函数来获取 Numba 实际使用的参数
def get_numba_params():
    # 实际使用的参数是在模块加载时确定的
    # 它们就是用于装饰器参数的值
    return {
        "nopython": nopython,
        "cache": cache,
        "enable64": enable64,
        "signal": signal,
    }
