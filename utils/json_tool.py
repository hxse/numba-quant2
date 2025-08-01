from pathlib import Path
import json


numba_params_path = "utils/numba_params_temp.json"


def read_numba_config(path=numba_params_path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            numba_params = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        numba_params = {"mode": "njit", "cache": True, "enable64": True}
        print(f"警告: 无法加载 {numba_params_path}，使用默认 Numba 配置。")
    return numba_params


def write_numba_config(
    path=numba_params_path,
    mode: str = "njit",
    cache: bool = True,
    enable64: bool = True,
):
    """将 Numba 配置写入临时 JSON 文件。"""
    config_to_write = {"mode": mode, "cache": cache, "enable64": enable64}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config_to_write, file, ensure_ascii=False, indent=4)


def delete_numba_config(path=numba_params_path):
    Path(path).unlink(missing_ok=True)


def load_numba_config(
    mode: str = "njit",
    cache: bool = True,
    enable64: bool = True,
):
    write_numba_config(mode=mode, cache=cache, enable64=enable64)

    from utils.numba_params import nb_params

    delete_numba_config()
    return nb_params
