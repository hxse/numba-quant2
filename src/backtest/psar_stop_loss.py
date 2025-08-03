import numba as nb
import numpy as np
import math

from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from src.indicators.psar import (
    psar_init,
    psar_update,
    PsarState,
)  # 导入 PsarState 类型和 psar_init 函数


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


# 定义新函数的签名
# 返回值：(triggered_by_psar: bool, trigger_price: float, new_psar_state: PsarState, psar_display_price: float)
# 注意：这里 psar_display_price 是为了可视化 psar 曲线，非交易使用
signature = nb.types.Tuple(
    (nb_bool_type, nb_float_type, PsarState, nb_float_type)
)(
    PsarState,  # psar_state (prev_psar_state)
    nb_int_type,  # current_direction (当前仓位方向，1多头，-1空头，0无仓位)
    nb_float_type,  # current_high
    nb_float_type,  # current_low
    nb_float_type,  # prev_high
    nb_float_type,  # prev_low
    nb_float_type,  # af0
    nb_float_type,  # af_step
    nb_float_type,  # max_af
    nb_int_type,  # init_idx_offset (用于 psar_init 的索引偏移量，因为传入的 high/low 可能是切片)
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def apply_psar_stop_loss(
    psar_state,
    current_direction,
    current_high,
    current_low,
    prev_high,
    prev_low,
    af0,
    af_step,
    max_af,
    init_idx_offset,  # 用于 psar_init 的索引偏移量，代表 K 线的绝对索引
):
    """
    独立封装 PSAR 跟踪止损逻辑。

    参数:
        psar_state: 上一 K 线的 PSAR 状态 (is_long, current_psar, current_ep, current_af)。
        current_direction: 当前 K 线开始时的持仓方向 (1: 多头, -1: 空头, 0: 无仓位)。
        current_high, current_low: 当前 K 线的高低价。
        prev_high, prev_low: 上一 K 线的高低价。
        af0, af_step, max_af: PSAR 参数。
        init_idx_offset: 当前 K 线在完整 OHLCV 数组中的绝对索引。
                         用于 psar_init 获取正确的历史 K 线。
        full_high_arr, full_low_arr, full_close_arr: 完整的 K 线数据数组，
                                                      用于在开仓时初始化 PSAR。

    返回:
        (triggered_by_psar: bool,      # 是否被 PSAR 止损触发
         trigger_price: float,         # 止损触发价格 (如果触发)
         new_psar_state: PsarState,    # 更新后的 PSAR 状态
         psar_display_price: float)    # 用于绘图的 PSAR 值 (当前有效方向上的 PSAR)
    """
    triggered_by_psar = False
    trigger_price = np.nan
    psar_display_price = np.nan  # 默认是 NaN

    new_psar_state = psar_state  # 默认保持不变

    # 如果有持仓，且 PSAR 状态有效，则更新 PSAR 并检查止损
    if current_direction != 0 and not math.isnan(psar_state[1]):
        is_long, current_psar, current_ep, current_af = psar_state

        # 使用 psar_update 预测本K线结束后的PSAR状态
        (
            updated_is_long,
            updated_psar,
            updated_ep,
            updated_af,
            psar_long_val,  # 当前预测的PSAR多头值
            psar_short_val,  # 当前预测的PSAR空头值
            reversal_val,
        ) = psar_update(
            psar_state,
            current_high,
            current_low,
            prev_high,
            prev_low,
            af_step,
            max_af,
        )

        # 更新 PSAR 状态
        new_psar_state = (updated_is_long, updated_psar, updated_ep, updated_af)

        # 设置用于显示的 PSAR 价格
        psar_display_price = psar_long_val if updated_is_long else psar_short_val

        # 检查穿透以触发离场
        if current_direction == 1:  # 当前持有多头仓位
            if current_low < psar_long_val:  # 当前K线低点跌破PSAR多头值
                triggered_by_psar = True
                trigger_price = psar_long_val  # 止损价为PSAR值
                # 止损后，PSAR状态应该重置，psar_display_price也应该清空
                new_psar_state = (False, np.nan, np.nan, np.nan)
                psar_display_price = np.nan

        elif current_direction == -1:  # 当前持有空头仓位
            if current_high > psar_short_val:  # 当前K线高点突破PSAR空头值
                triggered_by_psar = True
                trigger_price = psar_short_val  # 止损价为PSAR值
                # 止损后，PSAR状态应该重置，psar_display_price也应该清空
                new_psar_state = (False, np.nan, np.nan, np.nan)
                psar_display_price = np.nan

    # 如果没有持仓，或者PSAR止损触发了，PSAR状态应该被重置为无效
    elif current_direction == 0 and not math.isnan(psar_state[1]):
        # 如果当前无仓位，且 PSAR 状态是有效的 (例如，刚平仓但上一个循环还没有重置)，则重置
        new_psar_state = (False, np.nan, np.nan, np.nan)
        psar_display_price = np.nan

    # 返回结果
    return triggered_by_psar, trigger_price, new_psar_state, psar_display_price
