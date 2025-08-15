import numba as nb
import numpy as np
from .tr import calculate_tr
from .rma import calculate_rma
from utils.data_types import (
    get_indicator_params_child,
    get_indicator_result_child,
    loop_indicators_signature,
)

from .indicators_tool import check_bounds

from enum import Enum


atr_id = 3
atr_name = "atr"

atr_spec = {
    "id": atr_id,
    "name": atr_name,
    "ori_name": atr_name,
    "result_name": ["atr"],
    "default_params": [14],
    "param_count": 1,
    "result_count": 1,
    "temp_count": 1,
}


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


signature = nb.void(
    nb_float_type[:],
    nb_float_type[:],
    nb_float_type[:],
    nb_int_type,
    nb_float_type[:],
    nb_float_type[:],
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calculate_atr(high, low, close, period, atr_result, tr_result):
    # 越界检查
    if check_bounds(close, period, atr_result) == 0:
        return

    # 这里不需要初始化 result 为 NaN，因为 rma_func 应该负责填充
    # tr_result 也由 tr_func 填充，所以这里的初始化也可以移除，但为了安全保留
    atr_result[:] = np.nan
    tr_result[:] = np.nan

    # 计算真实范围 (TR) 值
    # tr_func 会将 TR 值填充到 tr_result 数组中，并在 tr_result[0] 处留下 NaN
    calculate_tr(high, low, close, tr_result)

    # 计算真实范围的移动平均 (RMA)
    # rma_func 应该将 RMA 值填充到 result 数组中。
    # RMA 应该在 period - 1 个 NaN 之后开始计算有效值。
    calculate_rma(tr_result, period, atr_result)


signature = nb.void(
    *loop_indicators_signature(nb_int_type, nb_float_type, nb_bool_type)
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calculate_atr_wrapper(
    _id, tohlcv, indicator_params_child, indicator_result_child, float_temp_array_child
):
    time = tohlcv[:, 0]
    open = tohlcv[:, 1]
    high = tohlcv[:, 2]
    low = tohlcv[:, 3]
    close = tohlcv[:, 4]
    volume = tohlcv[:, 5]

    atr_indicator_params_child = indicator_params_child[_id]
    atr_indicator_result_child = indicator_result_child[_id]

    if atr_indicator_params_child.shape[0] >= 1:
        atr_period = atr_indicator_params_child[0]

    if atr_indicator_result_child.shape[1] >= 1:
        atr_result = atr_indicator_result_child[:, 0]

    temp_arr_0 = float_temp_array_child[:, 0]
    temp_arr_0[:] = np.nan

    # atr_period 不用显示转换类型, numba会隐式把小数截断成整数(小数部分丢弃)
    calculate_atr(high, low, close, atr_period, atr_result, temp_arr_0)
