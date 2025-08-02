import numba as nb
import numpy as np
from utils.data_types import (
    get_indicator_params_child,
    get_indicator_result_child,
    get_indicator_wrapper_signal,
)

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


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


signature = nb.void(nb_float_type[:], nb_int_type, nb_float_type[:])


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calculate_sma(close, period, sma_result):
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


signature = nb.void(
    *get_indicator_wrapper_signal(nb_int_type, nb_float_type, nb_bool_type)
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calculate_sma_wrapper(
    tohlcv, indicator_params_child, indicator_result_child, float_temp_array_child, _id
):
    close = tohlcv[:, 4]

    sma_indicator_params_child = indicator_params_child[_id]
    sma_indicator_result_child = indicator_result_child[_id]

    sma_period = sma_indicator_params_child[0]
    sma_result = sma_indicator_result_child[:, 0]

    # sma_period 不用显示转换类型, numba会隐式把小数截断成整数(小数部分丢弃)
    calculate_sma(close, sma_period, sma_result)
