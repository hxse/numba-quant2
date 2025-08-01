# main_app.py
import typer
from numba import float64, float32

# 全局变量，用于保存 Typer 解析后的参数
# 初始化为 None 或默认值
numba_config_from_cli = {
    "nopython": False,
    "cache": False,
    "enable64": False,
    "signal": None,
}


def main_command(
    x_val: float,
    y_val: float,
    nopython: bool = False,
    cache: bool = False,
    enable64: bool = False,
):
    # 将 Typer 解析到的参数更新到全局变量
    numba_config_from_cli["nopython"] = nopython
    numba_config_from_cli["cache"] = cache
    numba_config_from_cli["enable64"] = enable64

    if enable64:
        numba_config_from_cli["signal"] = float64(float64, float64)
    else:
        numba_config_from_cli["signal"] = float32(float32, float32)

    print("entry", numba_config_from_cli["enable64"])

    # 现在才导入 numba_functions，确保它能看到更新后的全局变量
    from numba_cache_example3_another import add_with_offset_numba, get_numba_params

    # 获取 Numba 实际使用的参数（包括从 JSON 或默认值回退）
    final_numba_params = get_numba_params()
    print("final_numba_params", final_numba_params)

    result = add_with_offset_numba(x_val, y_val)

    print(f"计算结果: {result}")
    print(f"使用的 nopython: {final_numba_params['nopython']}")
    print(f"使用的 cache: {final_numba_params['cache']}")
    print(f"函数是否已 JIT 编译: {hasattr(add_with_offset_numba, '__numba__')}")


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_show_locals=False)
    app.command()(main_command)
    app()
