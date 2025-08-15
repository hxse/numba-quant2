from pathlib import Path
import json


numba_params_path = "utils/numba_params_temp.json"

default_mode = "njit"
default_cache = True
default_enable64 = True
default_max_registers = 24


def read_numba_config(path=numba_params_path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            numba_params = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        numba_params = {
            "mode": default_mode,
            "cache": default_cache,
            "enable64": default_enable64,
            "max_registers": default_max_registers,
        }
        print(f"警告: 无法加载 {numba_params_path}，使用默认 Numba 配置。")
    return numba_params


def write_numba_config(
    path=numba_params_path,
    mode: str = default_mode,
    cache: bool = default_cache,
    enable64: bool = default_enable64,
    max_registers: int = default_max_registers,
):
    """将 Numba 配置写入临时 JSON 文件。"""
    config_to_write = {
        "mode": mode,
        "cache": cache,
        "enable64": enable64,
        "max_registers": max_registers,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config_to_write, file, ensure_ascii=False, indent=4)


def delete_numba_config(path=numba_params_path):
    Path(path).unlink(missing_ok=True)


def load_numba_config(
    mode: str = default_mode,
    cache: bool = default_cache,
    enable64: bool = default_enable64,
    max_registers: int = default_max_registers,
):
    write_numba_config(
        mode=mode, cache=cache, enable64=enable64, max_registers=max_registers
    )

    from utils.numba_params import nb_params

    delete_numba_config()
    return nb_params
