import numba as nb
import numpy as np
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types, get_signature_child

sma_id = 0

sma_spec = {
    "id": sma_id,  # 指标id，不能跟其他指标重复。
    "name": "sma",  # 指标名
    "result_name": ["sma"],  # 结果数组列名
    "param_count": 1,  # 需要多少参数。
    "result_count": 1,  # 需要多少结果数组。
    "temp_count": 0,  # 该指标需要多少临时数组。
}


def calculate_sma(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    signature = nb.void(nb_float_type[:], nb_int_type, nb_float_type[:])

    def _calculate_sma(close, sma_period, sma_result):
        # 确保结果数组的长度与输入数据数组相同
        # 对于前 period - 1 个元素，填充 np.nan
        for i in range(min(sma_period - 1, len(close))):  # 避免数据长度不足越界
            sma_result[i] = np.nan

        # 从 period - 1 索引开始计算 SMA
        for i in range(len(close) - sma_period + 1):
            sum_val = 0.0
            for j in range(sma_period):
                sum_val += close[i + j]
            sma_result[i + sma_period - 1] = sum_val / sma_period

    return numba_wrapper(mode, signature=signature,
                         cache_enabled=cache)(_calculate_sma)


def calculate_sma_wrapper(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    signature = nb.void(nb_float_type[:, :], nb_float_type[:],
                        nb_float_type[:, :])

    _calculate_sma = calculate_sma(mode, cache=cache, dtype_dict=dtype_dict)

    def _calculate_sma_wrapper(micro_tohlcv, micro_indicator_params_child,
                               micro_indicator_result_child):

        time = micro_tohlcv[:, 0]
        open = micro_tohlcv[:, 1]
        high = micro_tohlcv[:, 2]
        low = micro_tohlcv[:, 3]
        close = micro_tohlcv[:, 4]
        volume = micro_tohlcv[:, 5]

        sma_period = micro_indicator_params_child[sma_period_idx]

        sma_result = micro_indicator_result_child[:, sma_result_idx]

        # sma_period 不用显示转换类型, numba会隐式把小数截断成整数(小数部分丢弃)
        _calculate_sma(close, sma_period, sma_result)

    return numba_wrapper(mode, signature=signature,
                         cache_enabled=cache)(_calculate_sma_wrapper)
