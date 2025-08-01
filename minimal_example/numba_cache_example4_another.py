import json
import os
from numba import jit, njit, cuda, float64, float32
# from numba import cuda # 如果有CUDA环境，请取消注释

# 配置文件路径
_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "numba_params_temp.json")

# --- 在模块加载时读取配置 ---
try:
    with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
        _numba_runtime_config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    _numba_runtime_config = {"mode": "njit", "cache": True, "enable64": True}
    print(f"警告: 无法加载 {_CONFIG_FILE}，使用默认 Numba 配置。")

# 提取配置参数
_mode = _numba_runtime_config.get("mode", "njit")
_cache = _numba_runtime_config.get("cache", True)
_enable64 = _numba_runtime_config.get("enable64", True)

# 根据 enable64 动态设置 signal 类型
_signal_type = float64(float64, float64) if _enable64 else float32(float32, float32)

print(
    f"[numba_functions] Loaded Numba config: mode={_mode}, enable64={_enable64}, cache={_cache}"
)


# --- 核心：根据mode返回不同的装饰器 ---
def numba_decorator(signal, mode, cache):
    if mode == "normal":
        print(f"[Numba Decorator] Applying normal JIT (nopython=False, cache={cache})")
        return jit(signal, nopython=False, cache=cache)
    elif mode == "njit":
        print(f"[Numba Decorator] Applying nopython JIT (nopython=True, cache={cache})")
        return njit(signal, cache=cache)
    elif mode == "cuda":
        return cuda.jit(signal, cache=cache)
    else:
        raise ValueError(f"未知 Numba 模式: {mode}")


# --- 应用动态选择的装饰器 ---
@numba_decorator(_signal_type, _mode, _cache)
def add2_numba(a, b):
    return a + b


@numba_decorator(_signal_type, _mode, _cache)
def add_with_offset_numba(a, b):
    return add2_numba(a, b) + 1


# --- 提供一个函数来获取实际使用的 Numba 参数 ---
def get_numba_params():
    return {
        "mode": _mode,
        "cache": _cache,
        "enable64": _enable64,
        "signal": _signal_type,
    }
