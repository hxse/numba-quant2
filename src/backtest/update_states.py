import numba as nb
import numpy as np
import math

from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

# 获取 Numba 数据类型
dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_float_type = dtype_dict["nb"]["float"]
nb_int_type = dtype_dict["nb"]["int"]
nb_bool_type = dtype_dict["nb"]["bool"]


# --- 辅助函数签名 ---
signature = nb.types.Tuple(
    (
        nb_float_type,  # entry_price
        nb_float_type,  # current_atr_tsl_price
        nb_int_type,  # current_direction (下一K线默认方向)
        nb_int_type,  # trade_status (下一K线默认状态)
    )
)(
    nb_int_type,  # current_k_idx
    nb_int_type,  # trade_status_prev
    nb_int_type,  # current_direction_prev
    nb_float_type[:],  # open_arr
    nb_float_type[:],  # atr_price_result
    nb_float_type,  # ATR_TSL_MULTIPLIER
    nb_float_type,  # prev_entry_price
    nb_float_type,  # prev_current_atr_tsl_price
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def update_initial_states_for_k_line(  # 函数名略微调整，去掉下划线以保持一致性
    current_k_idx,
    trade_status_prev,
    current_direction_prev,
    open_arr,
    atr_price_result,
    ATR_TSL_MULTIPLIER,
    prev_entry_price,
    prev_current_atr_tsl_price,
):
    """
    处理每根 K 线开始时的状态继承和入场价/ATR TSL 价格的更新。
    返回更新后的 entry_price, current_atr_tsl_price, current_direction, trade_status。
    """
    entry_price = prev_entry_price
    current_atr_tsl_price = prev_current_atr_tsl_price
    current_direction = current_direction_prev
    trade_status = trade_status_prev

    # 定义集合，用于归纳当前 K 线结束时的简化仓位方向
    IS_LONG_POSITION = (1, 2, 4)  # 多头进场, 持续多头, 平空开多
    IS_SHORT_POSITION = (-1, -2, -4)  # 空头进场, 持续空头, 平多开空
    IS_NO_POSITION = (0, 3, -3)  # 持续无仓位, 平多离场, 平空离场

    # 如果上一根 K 线是入场点，更新 entry_price 和 TSL 初始化
    if (
        trade_status_prev == 1
        or trade_status_prev == -1
        or trade_status_prev == 4
        or trade_status_prev == -4
    ):
        entry_price = open_arr[current_k_idx]  # 新开仓点为当前K线的开盘价
        # 开仓时，初始化 ATR TSL price
        if current_direction_prev == 1:  # 多头
            current_atr_tsl_price = (
                entry_price - atr_price_result[current_k_idx] * ATR_TSL_MULTIPLIER
            )
        elif current_direction_prev == -1:  # 空头
            current_atr_tsl_price = (
                entry_price + atr_price_result[current_k_idx] * ATR_TSL_MULTIPLIER
            )
        else:  # 理论上不会走到这里，但以防万一
            current_atr_tsl_price = np.nan

    # 如果上一根 K 线是平仓点，清空 entry_price 和 current_atr_tsl_price
    elif trade_status_prev == 3 or trade_status_prev == -3:
        entry_price = np.nan
        current_atr_tsl_price = np.nan

    # 默认情况下，继承上一K线的状态，如果无仓位则为0
    if trade_status_prev in IS_LONG_POSITION:
        trade_status = 2
        current_direction = 1
    elif trade_status_prev in IS_SHORT_POSITION:
        trade_status = -2
        current_direction = -1
    else:
        trade_status = 0
        current_direction = 0

    return entry_price, current_atr_tsl_price, current_direction, trade_status
