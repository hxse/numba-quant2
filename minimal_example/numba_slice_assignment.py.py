import numpy as np
from numba import jit


@jit(nopython=True)
def compare_arrays(array1, array2, output):
    # 1. 计算比较结果，得到一个临时布尔数组
    temp_result = array1 == array2

    # 2. 将临时结果切片赋值给 output 数组
    #    这里 output 的前几位被 temp_result 的值覆盖
    output[: len(temp_result)] = temp_result

    return output


# 准备数据
array1 = np.array([1, 2, 3])
array2 = np.array([1, 4, 3])

# 预先分配一个长度为 5 的数组，但它不会被使用
output = np.zeros(5, dtype=bool)

# 调用函数
result = compare_arrays(array1, array2, output)

print("调用函数后的结果数组的长度:", len(result))
print("预分配数组的长度:", len(output))
print("结果:", result)
print("结果:", output)
