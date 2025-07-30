import numba as nb
import numpy as np
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types
from enum import Enum

sma_id = 0
sma_name = "sma"
sma2_id = 1
sma2_name = "sma2"

sma_spec = {
    "id": sma_id,
    "name": sma_name,
    "ori_name": sma_name,
    "result_name": ["sma"],
    "default_params": [14],
    "param_count": 1,
    "result_count": 1,
    "temp_count": 0,
}
sma2_spec = {**sma_spec, "id": sma2_id, "name": sma2_name}


def calculate_sma(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    signature = nb.void(nb_float_type[:], nb_int_type, nb_float_type[:])

    def _calculate_sma(close, period, sma_result):
        data_length = len(close)
        result_length = len(sma_result)

        # 越界检查
        if result_length < data_length:
            return

        # 越界检查
        if period <= 0 or data_length < period or result_length < period:
            return

        for i in range(period - 1):
            sma_result[i] = np.nan

        for i in range(data_length - period + 1):
            sum_val = 0.0
            for j in range(period):
                sum_val += close[i + j]

            sma_result[i + period - 1] = sum_val / period

    return numba_wrapper(mode, signature=signature, cache_enabled=cache)(_calculate_sma)


def calculate_sma_wrapper(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    signature = nb.void(
        nb_float_type[:, :],  # tohlcv
        nb.types.Tuple(
            (nb_float_type[:], nb_float_type[:], nb_float_type[:])
        ),  # indicator_params_child
        nb.types.Tuple(
            (nb_float_type[:, :], nb_float_type[:, :], nb_float_type[:, :])
        ),  # indicator_result_child
        nb_int_type,  # _id
    )

    _calculate_sma = calculate_sma(mode, cache=cache, dtype_dict=dtype_dict)

    def _calculate_sma_wrapper(
        tohlcv, indicator_params_child, indicator_result_child, _id
    ):
        close = tohlcv[:, 4]

        sma_indicator_params_child = indicator_params_child[_id]
        sma_indicator_result_child = indicator_result_child[_id]

        sma_period = sma_indicator_params_child[0]
        sma_result = sma_indicator_result_child[:, 0]

        # sma_period 不用显示转换类型, numba会隐式把小数截断成整数(小数部分丢弃)
        _calculate_sma(close, sma_period, sma_result)

    return numba_wrapper(mode, signature=signature, cache_enabled=cache)(
        _calculate_sma_wrapper
    )
