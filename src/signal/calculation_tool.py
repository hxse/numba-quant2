import numba as nb
import numpy as np
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types
from enum import IntEnum, auto


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


def bool_compare(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    nb_bool_type = dtype_dict["nb"]["bool"]

    signature = nb.void(
        nb_float_type[:],  # array1
        nb_float_type[:],  # array2
        nb_bool_type[:],  # output
        nb_int_type,  # comparison_mode
        nb_int_type,  # assign_mode
    )

    def _bool_compare(array1, array2, output, comparison_mode, assign_mode):
        if (
            len(array1) != len(array2)
            or len(output) < len(array1)
            or len(output) < len(array2)
        ):
            return
        if assign_mode == AssignOperator.ASSIGN:
            for i in range(len(array1)):
                if comparison_mode == ComparisonOperator.eq:
                    output[i] = array1[i] == array2[i]
                elif comparison_mode == ComparisonOperator.ne:
                    output[i] = array1[i] != array2[i]
                elif comparison_mode == ComparisonOperator.gt:
                    output[i] = array1[i] > array2[i]
                elif comparison_mode == ComparisonOperator.ge:
                    output[i] = array1[i] >= array2[i]
                elif comparison_mode == ComparisonOperator.lt:
                    output[i] = array1[i] < array2[i]
                elif comparison_mode == ComparisonOperator.le:
                    output[i] = array1[i] <= array2[i]
        elif assign_mode == AssignOperator.BITWISE_AND:
            for i in range(len(array1)):
                if comparison_mode == ComparisonOperator.eq:
                    output[i] = output[i] & (array1[i] == array2[i])
                elif comparison_mode == ComparisonOperator.ne:
                    output[i] = output[i] & (array1[i] != array2[i])
                elif comparison_mode == ComparisonOperator.gt:
                    output[i] = output[i] & (array1[i] > array2[i])
                elif comparison_mode == ComparisonOperator.ge:
                    output[i] = output[i] & (array1[i] >= array2[i])
                elif comparison_mode == ComparisonOperator.lt:
                    output[i] = output[i] & (array1[i] < array2[i])
                elif comparison_mode == ComparisonOperator.le:
                    output[i] = output[i] & (array1[i] <= array2[i])
        elif assign_mode == AssignOperator.BITWISE_OR:
            for i in range(len(array1)):
                if comparison_mode == ComparisonOperator.eq:
                    output[i] = output[i] | (array1[i] == array2[i])
                elif comparison_mode == ComparisonOperator.ne:
                    output[i] = output[i] | (array1[i] != array2[i])
                elif comparison_mode == ComparisonOperator.gt:
                    output[i] = output[i] | (array1[i] > array2[i])
                elif comparison_mode == ComparisonOperator.ge:
                    output[i] = output[i] | (array1[i] >= array2[i])
                elif comparison_mode == ComparisonOperator.lt:
                    output[i] = output[i] | (array1[i] < array2[i])
                elif comparison_mode == ComparisonOperator.le:
                    output[i] = output[i] | (array1[i] <= array2[i])

    return numba_wrapper(mode, signature=signature, cache_enabled=cache)(_bool_compare)


def assign_elementwise(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    nb_bool_type = dtype_dict["nb"]["bool"]

    signature = nb.void(
        nb_bool_type[:],  # array
        nb_bool_type[:],  # output
        nb_int_type,  # mode
    )

    def _assign_elementwise(array, output, opt_mode):
        if len(output) < len(array):
            return
        for i in range(len(array)):
            if opt_mode == AssignOperator.ASSIGN:
                output[i] = array[i]
            elif opt_mode == AssignOperator.BITWISE_AND:
                output[i] = output[i] & array[i]
            elif opt_mode == AssignOperator.BITWISE_OR:
                output[i] = output[i] | array[i]

    return numba_wrapper(mode, signature=signature, cache_enabled=cache)(
        _assign_elementwise
    )
