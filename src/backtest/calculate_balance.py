# exit_signals.py

import numba as nb
import numpy as np
from src.indicators.psar import psar_init, psar_update

import math
from utils.data_types import get_params_child_signature


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper


dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]

# 定义 Numba 签名
signature = nb.void(
    nb_int_type,  # i
    nb_int_type,  # last_i
    nb_float_type,  # target_price
    nb_float_type[:, :],  # tohlcv
    nb_float_type[:],  # position_status_result
    nb_float_type[:],  # entry_price_result
    nb_float_type[:],  # exit_price_result
    nb_float_type[:],  # equity_result
    nb_float_type[:],  # balance_result
    nb_float_type[:],  # drawdown_result
    nb_float_type[:],  # temp_max_balance_array
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_LONG_POSITION
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_SHORT_POSITION
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_NO_POSITION
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calc_balance(
    i,
    last_i,
    target_price,
    tohlcv,
    position_status_result,
    entry_price_result,
    exit_price_result,
    equity_result,
    balance_result,
    drawdown_result,
    temp_max_balance_array,
    IS_LONG_POSITION,
    IS_SHORT_POSITION,
    IS_NO_POSITION,
):
    # 6. 从 tohlcv 中提取时间、开盘、最高、最低、收盘、成交量数组
    time_arr = tohlcv[:, 0]
    open_arr = tohlcv[:, 1]
    high_arr = tohlcv[:, 2]
    low_arr = tohlcv[:, 3]
    close_arr = tohlcv[:, 4]
    volume_arr = tohlcv[:, 5]

    # equity_result, balance_result, drawdown_result, temp_max_balance_array,
    # 这四个数组在循环前就被初始化为初始本金
    balance_result[i] = balance_result[last_i]
    equity_result[i] = balance_result[i]

    # 2. 处理平仓和反手，更新 balance
    if (
        position_status_result[i] in (3, -4)
        and position_status_result[last_i] in IS_LONG_POSITION
    ):
        profit_percentage = (
            exit_price_result[i] - entry_price_result[last_i]
        ) / entry_price_result[last_i]
        balance_result[i] = balance_result[last_i] * (1 + profit_percentage)
        equity_result[i] = balance_result[i]

    elif (
        position_status_result[i] in (-3, 4)
        and position_status_result[last_i] in IS_SHORT_POSITION
    ):
        # 修正：空头平仓的利润计算
        profit_percentage = (
            entry_price_result[last_i] - exit_price_result[i]
        ) / entry_price_result[last_i]
        balance_result[i] = balance_result[last_i] * (1 + profit_percentage)
        equity_result[i] = balance_result[i]

    # 3. 处理持仓状态，计算浮盈并更新 equity
    elif position_status_result[i] == 2:  # 多头持仓
        profit_percentage = (
            open_arr[i] - entry_price_result[last_i]
        ) / entry_price_result[last_i]
        equity_result[i] = balance_result[last_i] * (1 + profit_percentage)

    elif position_status_result[i] == -2:  # 空头持仓
        # 修正：空头持仓的浮盈计算
        profit_percentage = (
            entry_price_result[last_i] - open_arr[i]
        ) / entry_price_result[last_i]
        equity_result[i] = balance_result[last_i] * (1 + profit_percentage)

    temp_max_balance_array[i] = max(temp_max_balance_array[last_i], balance_result[i])

    if temp_max_balance_array[i] > 0:
        # 修正：回撤计算公式
        drawdown_result[i] = (
            temp_max_balance_array[i] - balance_result[i]
        ) / temp_max_balance_array[i]
