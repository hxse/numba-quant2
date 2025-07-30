import numpy as np
from numba import njit, float64, int64


# 1. 定义几个具有相同签名的 Numba JIT 函数
@njit(float64(float64, float64))
def add_func(a, b):
    return a + b


@njit(float64(float64, float64))
def subtract_func(a, b):
    return a - b


@njit(float64(float64, float64))
def multiply_func(a, b):
    return a * b


@njit(float64(float64, float64))
def divide_func(a, b):
    # 避免除以零错误
    if b == 0:
        return np.nan
    return a / b


# 2. 将这些 Numba JIT 函数放入一个等长的二维元组
GLOBAL_EQUAL_LENGTH_JIT_FUNCS = (
    (add_func, subtract_func),  # 长度为 2 的元组
    (multiply_func, divide_func),  # 长度为 2 的元组
    (add_func, multiply_func),  # 长度为 2 的元组
)


# 3. 定义一个在 JIT 中尝试调用这些函数的函数
@njit(float64(int64, int64, float64, float64))
def execute_equal_length_nested_jit_function(outer_idx, inner_idx, x, y):
    """
    通过嵌套索引调用 GLOBAL_EQUAL_LENGTH_JIT_FUNCS 中的 Numba JIT 函数。
    如果索引越界，Numba 会自动抛出 IndexError。
    """
    # 直接访问，让 Numba 自身处理索引越界（它会抛出 IndexError）
    selected_func = GLOBAL_EQUAL_LENGTH_JIT_FUNCS[outer_idx][inner_idx]
    return selected_func(x, y)


# --- 测试部分 ---
if __name__ == "__main__":
    print("--- 尝试用等长二维元组调用 JIT 函数 (无内部边界检查) ---")

    # 成功调用的用例
    test_cases_success = [
        (0, 0, 10.0, 5.0, "add_func"),
        (0, 1, 10.0, 5.0, "subtract_func"),
        (1, 0, 10.0, 5.0, "multiply_func"),
        (1, 1, 10.0, 5.0, "divide_func"),
        (2, 0, 100.0, 20.0, "add_func (重复)"),
    ]

    for outer, inner, x_val, y_val, desc in test_cases_success:
        try:
            result = execute_equal_length_nested_jit_function(
                outer, inner, x_val, y_val
            )
            print(f"调用 [{outer}][{inner}] ({desc}): {x_val} op {y_val} = {result}")
        except IndexError as e:
            print(f"调用 [{outer}][{inner}] ({desc}) 捕获到意外错误: {e}")
        except Exception as e:
            print(
                f"调用 [{outer}][{inner}] ({desc}) 捕获到其他意外错误: {type(e).__name__}: {e}"
            )

    print("\n--- 尝试越界索引，并在 __main__ 中捕获异常 ---")

    # 内部索引越界的用例
    try:
        result_oob_inner = execute_equal_length_nested_jit_function(0, 2, 10.0, 5.0)
        print(f"调用 [0][2] (内部越界): {result_oob_inner}")
    except IndexError as e:
        print(f"调用 [0][2] (内部越界) 捕获到预期错误: {e}")
    except Exception as e:
        print(f"调用 [0][2] (内部越界) 捕获到其他意外错误: {type(e).__name__}: {e}")

    # 外部索引越界的用例
    try:
        result_oob_outer = execute_equal_length_nested_jit_function(3, 0, 10.0, 5.0)
        print(f"调用 [3][0] (外部越界): {result_oob_outer}")
    except IndexError as e:
        print(f"调用 [3][0] (外部越界) 捕获到预期错误: {e}")
    except Exception as e:
        print(f"调用 [3][0] (外部越界) 捕获到其他意外错误: {type(e).__name__}: {e}")
