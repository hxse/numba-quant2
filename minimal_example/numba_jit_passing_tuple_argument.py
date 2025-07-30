import numba as nb
import numpy as np


# 1. 核心 JIT 函数：接受一个名为 'args' 的普通参数
# 签名：
# nb.float64 (返回值类型)
# nb.types.Tuple((nb.float64, nb.float64, nb.float64[:])) -> 这是 'args' 这个单一参数的完整类型
# 注意：移除了 UniTuple 包装！
@nb.jit(
    nb.float64(
        nb.types.Tuple(
            (nb.float64, nb.float64, nb.float64[:])
        )  # 直接定义 args 是一个Tuple
    ),
    nopython=True,
)
def my_jitted_function_single_arg_param_simplified(
    args,
):  # <--- 注意这里是 'args' 而不是 '*args'
    """
    一个接受名为 'args' 的单一参数 (元组) 的 Numba JIT 函数。
    签名清晰地指出 'args' 是一个 Tuple(...) 类型。
    函数内部直接从 args 解包出三个独立的值。
    """
    # 关键步骤：Numba 将外部传入的唯一元组参数视为 'args' 这个变量的值。
    # 根据新的简化签名，'args' 本身就是那个包含三个元素的元组。
    # 所以，可以直接解包。
    val1, val2, arr = args  # 直接从 args 解包

    # 执行一些操作
    result = val1 * val2
    for i in range(arr.shape[0]):
        arr[i] = arr[i] + result
    return arr[0]  # 返回修改后数组的第一个元素，以便测试


# 2. 调用 JIT 函数的 Python 代码
if __name__ == "__main__":
    # 定义数据
    val_a = 10.0
    val_b = 5.0
    array_c = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    print("--- 简化签名并直接传递元组的 JIT 示例 ---")

    # 在调用时，将所有希望传入 JIT 函数的参数打包成一个元组。
    # 这个元组将作为 my_jitted_function_single_arg_param_simplified 的唯一参数。
    parameters_as_inner_tuple = (val_a, val_b, array_c)

    # 直接将 parameters_as_inner_tuple 传递给 JIT 函数
    # 无需额外的 UniTuple 包装
    result = my_jitted_function_single_arg_param_simplified(parameters_as_inner_tuple)

    print(f"原始 array_c: {np.array([1.0, 2.0, 3.0], dtype=np.float64)}")
    print(f"JIT 函数返回值 (array_c[0] + val_a * val_b): {result}")
    print(f"修改后的 array_c: {array_c}")

    # 验证计算结果
    expected_result_first_element = 1.0 + (10.0 * 5.0)
    print(f"预期 array_c[0]: {expected_result_first_element}")
    assert result == expected_result_first_element
    assert array_c[0] == expected_result_first_element
    print("测试通过！🎉")
