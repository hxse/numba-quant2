import typer
import json
import os

# 定义临时配置文件路径，与 numba_functions.py 中一致
_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "numba_params_temp.json")


def write_numba_config(mode: str, cache: bool, enable64: bool):
    """将 Numba 配置写入临时 JSON 文件。"""
    config_to_write = {"mode": mode, "cache": cache, "enable64": enable64}
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_to_write, f, indent=4)
    print(
        f"[main_app] Numba config written to {_CONFIG_FILE}: mode={mode}, enable64={enable64}"
    )


def main_command(
    x_val: float,
    y_val: float,
    mode: str = typer.Option(
        "njit",
        help="Numba JIT mode: 'normal' (nopython=False), 'njit' (nopython=True), or 'cuda'",
    ),
    cache: bool = typer.Option(True, help="Enable caching for Numba"),
    enable64: bool = typer.Option(
        True, help="Use float64 for Numba types (otherwise float32)"
    ),
):
    # 步骤 1: 将 Typer 解析到的参数写入配置文件
    write_numba_config(mode, cache, enable64)

    # 步骤 2: 导入 Numba 函数模块。
    # 当 numba_functions.py 被导入时，它会读取刚刚更新的配置文件。
    from numba_cache_example4_another import add_with_offset_numba, get_numba_params

    # 步骤 3: 获取 Numba 实际使用的参数并执行计算
    final_numba_params = get_numba_params()

    print(f"main_app_command called with mode: {mode}, enable64: {enable64}")
    print("entry params", {"mode": mode, "cache": cache, "enable64": enable64})
    print("final params from Numba module:", final_numba_params)

    result = add_with_offset_numba(x_val, y_val)

    print(f"计算结果: {result}")
    print(f"使用的 mode: {final_numba_params['mode']}")
    print(f"使用的 cache: {final_numba_params['cache']}")
    print(f"使用的 enable64: {final_numba_params['enable64']}")
    print(f"函数是否已 JIT 编译: {hasattr(add_with_offset_numba, '__numba__')}")


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_show_locals=False)
    app.command()(main_command)
    app()
