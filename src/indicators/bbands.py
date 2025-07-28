import numba as nb
import numpy as np
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types
import math
from .sma import calculate_sma

bbands_id = 2
bbands_name = "bbands"

bbands_spec = {
    "id": bbands_id,  # 指标id，不能跟其他指标重复。
    "name": bbands_name,  # 指标名
    "ori_name": bbands_name,
    "result_name": ["middle_result", "upper_result", "lower_result"],  # 结果数组列名
    "default_params": [14, 2.0],
    "param_count": 2,  # 需要多少参数。
    "result_count": 3,  # 需要多少结果数组。
    "temp_count": 0,  # 该指标需要多少临时数组。
}


def calculate_bbands(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    signature = nb.void(nb_float_type[:], nb_int_type, nb_float_type,
                        nb_float_type[:], nb_float_type[:], nb_float_type[:])

    _calculate_sma = calculate_sma(mode, cache=cache, dtype_dict=dtype_dict)

    def _calculate_bbands(close, period, std_mult, middle_result, upper_result,
                          lower_result):
        data_length = len(close)
        result_length = len(middle_result)  # 假设所有结果数组长度相同

        # 越界检查
        if result_length < data_length:
            return

        # 越界检查
        if period <= 0 or data_length < period or result_length < period:
            return

        _calculate_sma(close, period, middle_result)

        # 对于前 period - 1 个元素，填充 np.nan
        for i in range(period - 1):
            upper_result[i] = np.nan
            lower_result[i] = np.nan

        # 从 period - 1 索引开始计算标准差和布林带
        for i in range(len(close) - period + 1):
            variance = 0.0
            # 注意：这里 i + period - 1 是当前 SMA 对应的原始数据索引
            for j in range(period):
                diff = close[i + j] - middle_result[i + period - 1]
                variance += diff * diff
            std = math.sqrt(variance / period)
            upper_result[i + period - 1] = (middle_result[i + period - 1] +
                                            std_mult * std)
            lower_result[i + period - 1] = (middle_result[i + period - 1] -
                                            std_mult * std)

    return numba_wrapper(mode, signature=signature,
                         cache_enabled=cache)(_calculate_bbands)


def calculate_bbands_wrapper(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    signature = nb.void(
        nb_float_type[:, :],  # tohlcv
        nb.types.Tuple((nb_float_type[:], nb_float_type[:],
                        nb_float_type[:])),  # indicator_params_child
        nb.types.Tuple((nb_float_type[:, :], nb_float_type[:, :],
                        nb_float_type[:, :])),  # indicator_result_child
        nb_int_type  # _id
    )

    _calculate_bbands = calculate_bbands(mode,
                                         cache=cache,
                                         dtype_dict=dtype_dict)

    def _calculate_bbands_wrapper(tohlcv, indicator_params_child,
                                  indicator_result_child, _id):

        time = tohlcv[:, 0]
        open = tohlcv[:, 1]
        high = tohlcv[:, 2]
        low = tohlcv[:, 3]
        close = tohlcv[:, 4]
        volume = tohlcv[:, 5]

        bbands_indicator_params_child = indicator_params_child[_id]
        bbands_indicator_result_child = indicator_result_child[_id]

        bbands_period = bbands_indicator_params_child[0]
        bbands_std_mult = bbands_indicator_params_child[1]

        middle_result = bbands_indicator_result_child[:, 0]
        upper_result = bbands_indicator_result_child[:, 1]
        lower_result = bbands_indicator_result_child[:, 2]

        # bbands_period 不用显示转换类型, numba会隐式把小数截断成整数(小数部分丢弃)
        _calculate_bbands(close, bbands_period, bbands_std_mult, middle_result,
                          upper_result, lower_result)

    return numba_wrapper(mode, signature=signature,
                         cache_enabled=cache)(_calculate_bbands_wrapper)
