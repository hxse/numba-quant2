import numba as nb
import numpy as np
from utils.data_types import loop_indicators_signature
from .indicators_tool import check_bounds
from utils.numba_utils import nb_wrapper


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types


dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


sma_spec = {
    "name": "sma",
    "ori_name": "sma",
    "result_name": ["sma"],
    "default_params": [14],
    "param_count": 1,
    "result_count": 1,
    "temp_count": 0,
}


sma2_spec = {
    **sma_spec,
    #  "id": sma2_id,
    "name": "sma2",
    "result_name": ["sma2"],
}


signature = nb.void(nb_float_type[:], nb_int_type, nb_float_type[:])


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calculate_sma(close, period, sma_result):
    # 越界检查
    if check_bounds(close, period, sma_result) == 0:
        return

    data_length = len(close)

    for i in range(period - 1):
        sma_result[i] = np.nan

    for i in range(data_length - period + 1):
        sum_val = 0.0
        for j in range(period):
            sum_val += close[i + j]

        sma_result[i + period - 1] = sum_val / period


signature = nb.void(
    *loop_indicators_signature(nb_int_type, nb_float_type, nb_bool_type)
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calculate_sma_wrapper(
    _id, tohlcv, indicator_params_child, indicator_result_child, float_temp_array_child
):
    close = tohlcv[:, 4]

    sma_indicator_params_child = indicator_params_child[_id]
    sma_indicator_result_child = indicator_result_child[_id]

    if sma_indicator_params_child.shape[0] >= 1:
        sma_period = sma_indicator_params_child[0]

    if sma_indicator_result_child.shape[0] >= 1:
        sma_result = sma_indicator_result_child[:, 0]

    # sma_period 不用显示转换类型, numba会隐式把小数截断成整数(小数部分丢弃)
    calculate_sma(close, sma_period, sma_result)
