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

from enum import IntEnum, auto


# 定义一个 Numba 枚举，用于表示 ATR 离场类型
class AtrExitType(IntEnum):
    NONE = 0
    SL = 1  # Stop Loss
    TP = 2  # Take Profit
    TSL = 3  # Trailing Stop Loss


# 定义新函数的签名
# 返回值：(triggered: bool, trigger_price: float, new_trailing_stop_price: float)
# new_trailing_stop_price 仅在 AtrExitType.TSL 时有效
signature = nb.types.Tuple((nb_bool_type, nb_float_type, nb_float_type))(
    nb_int_type,  # exit_type (AtrExitType 枚举值)
    nb_int_type,  # current_direction (当前仓位方向，1多头，-1空头)
    nb_float_type,  # current_close (当前 K 线的收盘价)
    nb_float_type,  # atr_value (当前 K 线的 ATR 值)
    nb_float_type,  # entry_price (入场价格)
    nb_float_type,  # atr_multiplier (ATR 乘数)
    nb_float_type,  # prev_trailing_stop_price (上一 K 线的跟踪止损价，仅用于TSL)
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def apply_atr_exit_logic(
    exit_type,
    current_direction,
    current_close,
    atr_value,
    entry_price,
    atr_multiplier,
    prev_trailing_stop_price,  # 用于 ATR TSL
):
    """
    封装 ATR 止损、止盈、跟踪止损的通用逻辑。

    参数:
        exit_type: AtrExitType 枚举值 (SL, TP, TSL)。
        current_direction: 当前持仓方向 (1: 多头, -1: 空头)。
        current_close: 当前 K 线的收盘价。
        atr_value: 当前 K 线的 ATR 值。
        entry_price: 入场价格。
        atr_multiplier: ATR 乘数 (用于计算止损止盈距离)。
        prev_trailing_stop_price: 上一 K 线的跟踪止损价格 (仅用于 ATR TSL)。

    返回:
        (triggered: bool,      # 是否被 ATR 逻辑触发
         trigger_price: float, # 触发价格 (如果触发)
         new_trailing_stop_price: float) # 更新后的跟踪止损价 (如果 exit_type 是 TSL)
    """
    triggered = False
    trigger_price = np.nan
    new_trailing_stop_price = prev_trailing_stop_price  # 默认不变

    if math.isnan(atr_value):  # 如果 ATR 值无效，直接返回
        return triggered, trigger_price, new_trailing_stop_price

    exit_distance = atr_value * atr_multiplier

    if current_direction == 1:  # 多头仓位
        if exit_type == AtrExitType.SL:  # ATR 止损 (SL)
            sl_price = entry_price - exit_distance
            if current_close <= sl_price:
                triggered = True
                trigger_price = sl_price
        elif exit_type == AtrExitType.TP:  # ATR 止盈 (TP)
            tp_price = entry_price + exit_distance
            if current_close >= tp_price:
                triggered = True
                trigger_price = tp_price
        elif exit_type == AtrExitType.TSL:  # ATR 跟踪止损 (TSL)
            if math.isnan(prev_trailing_stop_price):
                # 首次设置跟踪止损：入场价 - 止损距离
                new_trailing_stop_price = entry_price - exit_distance
            else:
                # 跟踪止损向上移动：当前收盘价 - 止损距离，但不低于前一个跟踪止损价
                new_trailing_stop_price = max(
                    prev_trailing_stop_price, current_close - exit_distance
                )

            if current_close <= new_trailing_stop_price:
                triggered = True
                trigger_price = new_trailing_stop_price

    elif current_direction == -1:  # 空头仓位
        if exit_type == AtrExitType.SL:  # ATR 止损 (SL)
            sl_price = entry_price + exit_distance
            if current_close >= sl_price:
                triggered = True
                trigger_price = sl_price
        elif exit_type == AtrExitType.TP:  # ATR 止盈 (TP)
            tp_price = entry_price - exit_distance
            if current_close <= tp_price:
                triggered = True
                trigger_price = tp_price
        elif exit_type == AtrExitType.TSL:  # ATR 跟踪止损 (TSL)
            if math.isnan(prev_trailing_stop_price):
                # 首次设置跟踪止损：入场价 + 止损距离
                new_trailing_stop_price = entry_price + exit_distance
            else:
                # 跟踪止损向下移动：当前收盘价 + 止损距离，但不高于前一个跟踪止损价
                new_trailing_stop_price = min(
                    prev_trailing_stop_price, current_close + exit_distance
                )

            if current_close >= new_trailing_stop_price:
                triggered = True
                trigger_price = new_trailing_stop_price

    return triggered, trigger_price, new_trailing_stop_price
