import numba as nb
import numpy as np
from enum import IntEnum, auto

from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


class ComparisonOperator(IntEnum):
    """
    定义各种比较运算符，包括区分大小写和不区分大小写（如果适用）。
    """

    # 等于 (Equals)
    eq = auto()  # 默认或常规等于

    # 不等于 (Not Equals)
    ne = auto()  # 默认或常规不等于

    # 大于 (Greater Than)
    gt = auto()

    # 大于或等于 (Greater Than or Equal)
    ge = auto()

    # 小于 (Less Than)
    lt = auto()

    # 小于或等于 (Less Than or Equal)
    le = auto()


class AssignOperator(IntEnum):
    """
    定义逐元素赋值操作的类型。
    """

    ASSIGN = auto()  # 直接赋值 (例如 1)
    BITWISE_AND = auto()  # 位与赋值 (例如 2)
    BITWISE_OR = auto()  # 位或赋值 (例如 3)


class TriggerOperator(IntEnum):
    """
    定义基于布尔条件的序列行为来触发一个操作的模式。
    """

    # 连续触发：只要条件为真，就持续触发。
    CONTINUOUS = auto()

    # 边缘触发：仅在布尔条件状态发生变化时触发。
    # 这个成员可以同时处理从 False -> True 和 True -> False 的两种情况。
    EDGE = auto()


signature = nb.void(
    nb_float_type[:],  # array1
    nb_float_type[:],  # array2
    nb_bool_type[:],  # output
    nb_bool_type[:],  # temp_array
    nb_int_type,  # comparison_mode
    nb_int_type,  # assign_mode
    nb_int_type,  # trigger_mode
)

if nb_params["mode"] in ["normal", "njit"]:

    @nb_wrapper(
        mode=nb_params["mode"],
        signature=signature,
        cache_enabled=nb_params.get("cache", True),
    )
    def bool_compare(
        array1, array2, output, temp_array, comparison_mode, assign_mode, trigger_mode
    ):
        if (
            len(array1) != len(array2)
            or len(output) < len(array1)
            or len(output) < len(array2)
        ):
            return

        # 计算大小比较
        if comparison_mode == ComparisonOperator.eq:
            temp_array[:] = array1 == array2
        elif comparison_mode == ComparisonOperator.ne:
            temp_array[:] = array1 != array2
        elif comparison_mode == ComparisonOperator.gt:
            temp_array[:] = array1 > array2
        elif comparison_mode == ComparisonOperator.ge:
            temp_array[:] = array1 >= array2
        elif comparison_mode == ComparisonOperator.lt:
            temp_array[:] = array1 < array2
        elif comparison_mode == ComparisonOperator.le:
            temp_array[:] = array1 <= array2

        # 计算完大小比较后,计算异或
        if assign_mode == AssignOperator.ASSIGN:
            output[:] = temp_array
        elif assign_mode == AssignOperator.BITWISE_AND:
            output[:] = output & temp_array
        elif assign_mode == AssignOperator.BITWISE_OR:
            output[:] = output | temp_array

        # 计算完大小比较和异或后,计算边缘触发
        if trigger_mode == TriggerOperator.CONTINUOUS:
            pass
        elif trigger_mode == TriggerOperator.EDGE:
            temp_array[1:] = output[:-1]
            temp_array[0] = False
            output[:] = output & ~temp_array

elif nb_params["mode"] in ["cuda"]:

    @nb_wrapper(
        mode=nb_params["mode"],
        signature=signature,
        cache_enabled=nb_params.get("cache", True),
    )
    def bool_compare(
        array1, array2, output, temp_array, comparison_mode, assign_mode, trigger_mode
    ):
        if (
            len(array1) != len(array2)
            or len(output) < len(array1)
            or len(output) < len(array2)
        ):
            return

        # 由于是在cuda设备函数内部,不存在并发,只能用循环线性处理
        for i in range(len(array1)):
            # 计算大小比较
            if comparison_mode == ComparisonOperator.eq:
                temp_array[i] = array1[i] == array2[i]
            elif comparison_mode == ComparisonOperator.ne:
                temp_array[i] = array1[i] != array2[i]
            elif comparison_mode == ComparisonOperator.gt:
                temp_array[i] = array1[i] > array2[i]
            elif comparison_mode == ComparisonOperator.ge:
                temp_array[i] = array1[i] >= array2[i]
            elif comparison_mode == ComparisonOperator.lt:
                temp_array[i] = array1[i] < array2[i]
            elif comparison_mode == ComparisonOperator.le:
                temp_array[i] = array1[i] <= array2[i]

            # 计算完大小比较后,计算异或
            if assign_mode == AssignOperator.ASSIGN:
                output[i] = temp_array[i]
            elif assign_mode == AssignOperator.BITWISE_AND:
                output[i] = output[i] & temp_array[i]
            elif assign_mode == AssignOperator.BITWISE_OR:
                output[i] = output[i] | temp_array[i]

            # 计算完大小比较和异或后,计算边缘触发
            if trigger_mode == TriggerOperator.CONTINUOUS:
                pass
            elif trigger_mode == TriggerOperator.EDGE:
                if i >= 1:
                    output[i] = output[i] & ~output[i - 1]
