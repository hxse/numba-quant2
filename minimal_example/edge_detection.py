import numpy as np
from numba import jit


@jit(nopython=True)
def rising_edge_detection_in_place(a, b, c):
    """
    用 Numba 和 NumPy 实现上升沿检测，并将结果写入预分配的数组 c。

    参数:
        a (np.ndarray): 第一个 NumPy 数组。
        b (np.ndarray): 第二个 NumPy 数组。
        c (np.ndarray): 预先分配好的布尔数组，用于存储结果。
                       其大小必须与 a 和 b 相同。
    """
    n = len(a)

    # 1. 计算 (a > b)，得到布尔数组 `is_a_greater`
    is_a_greater = a > b

    # 2. 模拟 .shift() 操作，创建一个新的临时数组
    is_a_greater_shifted = np.empty_like(is_a_greater)
    is_a_greater_shifted[1:] = is_a_greater[:-1]
    is_a_greater_shifted[0] = False

    # 3. 计算 !(a > b).shift()
    not_shifted = ~is_a_greater_shifted

    # 4. 结合两个条件，并将结果直接写入数组 c
    #    这个操作会修改传入的 c 数组
    c[:] = is_a_greater & not_shifted


# --- 示例数据 ---
a = np.array([10, 20, 25, 22, 30])
b = np.array([12, 18, 20, 24, 28])

# 预先分配结果数组 c
c = np.zeros(len(a), dtype=np.bool_)

# 调用 Numba 编译后的函数
rising_edge_detection_in_place(a, b, c)

# 打印结果
print("数组 a:", a)
print("数组 b:", b)
print("\n预分配的结果数组 c:", c)
# 预期的结果是 [False, False, True, False, True]
