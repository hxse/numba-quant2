import sys
from numba import jit, float64, float32
import typer
import json


with open("minimal_example/numba_params.json", "r", encoding="utf-8") as file:
    numba_params = json.load(file)


nopython = True if numba_params.get("nopython") else False
cache = True if numba_params.get("cache") else False
enable64 = True if numba_params.get("enable64") else False

signal = (
    float64(float64, float64)
    if numba_params.get("enable64")
    else float32(float32, float32)
)


def factory2(nopython: bool = True, cache: bool = True):
    @jit(signal, nopython=nopython, cache=cache)
    def _calc(a, b):
        return a + b

    return _calc


add2 = factory2(nopython=nopython, cache=cache)

_add2 = (add2,)


def factory(nopython: bool = True, cache: bool = True):
    @jit(signal, nopython=nopython, cache=cache)
    def _calc(a, b):
        return _add2[0](a, b) + 1

    return _calc


add = factory(nopython=nopython, cache=cache)


def main(x_val: float, y_val: float, nopython: bool = True, cache: bool = True):
    result = add(x_val, y_val)

    print(f"计算结果: {result}")
    print(f"使用的 nopython: {nopython}")
    print(f"使用的 cache: {cache}")

    # 检查是否已 JIT 编译
    # Numba 没有直接的 is_jitted 属性
    # 可以通过检查 __numba__ 属性是否存在来间接判断
    is_jitted = hasattr(add, "__numba__")
    print(f"函数是否已 JIT 编译: {is_jitted}")


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_show_locals=False)
    app.command(help="also ma")(main)
    app.command("ma", hidden=True)(main)
    app()
