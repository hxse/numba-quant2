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

# 导入 PSAR 初始化函数
from src.backtest.psar_initialization import initialize_psar_state


# --- 辅助函数签名 ---
signature = nb.types.Tuple(
    (
        nb_int_type,  # new_trade_status
        nb_float_type,  # new_trigger_price
        nb_int_type,  # new_current_direction
        nb.types.Tuple(
            (nb_bool_type, nb_float_type, nb_float_type, nb_float_type)
        ),  # new_psar_state
        nb_float_type,  # new_current_atr_tsl_price
    )
)(
    nb_int_type,  # current_k_idx
    nb_int_type,  # trade_status_prev
    nb_int_type,  # current_direction_prev
    nb_bool_type,  # enter_long_prev
    nb_bool_type,  # exit_long_prev
    nb_bool_type,  # enter_short_prev
    nb_bool_type,  # exit_short_prev
    nb_float_type[:],  # open_arr
    nb_float_type[:],  # high_arr
    nb_float_type[:],  # low_arr
    nb_float_type[:],  # close_arr
    nb_float_type[:],  # atr_price_result
    nb_float_type,  # af0
    nb_float_type,  # ATR_TSL_MULTIPLIER
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_LONG_POSITION
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_SHORT_POSITION
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_NO_POSITION
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def handle_entry_exit_signals(  # 函数名略微调整
    current_k_idx,
    trade_status_prev,
    current_direction_prev,
    enter_long_prev,
    exit_long_prev,
    enter_short_prev,
    exit_short_prev,
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    atr_price_result,
    af0,
    ATR_TSL_MULTIPLIER,
    IS_LONG_POSITION,
    IS_SHORT_POSITION,
    IS_NO_POSITION,
):
    """
    处理开仓、平仓和反转信号，并初始化 PSAR 和 ATR TSL 状态。
    返回更新后的 trade_status, trigger_price, current_direction, psar_state, current_atr_tsl_price。
    """
    new_trade_status = 0
    new_trigger_price = np.nan
    new_current_direction = 0
    new_psar_state = (False, np.nan, np.nan, np.nan)
    new_current_atr_tsl_price = np.nan

    # Convert tuples to Numba set for efficient `in` checks
    # Numba sets are usually better for `in` checks than iterating through tuples
    nb_is_long_set = set(IS_LONG_POSITION)
    nb_is_short_set = set(IS_SHORT_POSITION)
    nb_is_no_position_set = set(IS_NO_POSITION)

    # 优先处理开平仓反转 (平多开空或平空开多)
    # 平多进空 (-4)
    if (
        (
            trade_status_prev in nb_is_long_set
            or trade_status_prev in nb_is_no_position_set
        )
        and exit_long_prev
        and enter_short_prev
        and not enter_long_prev
        and not exit_short_prev
    ):
        new_trade_status = -4
        new_trigger_price = open_arr[current_k_idx]
        new_current_direction = -1
        new_psar_state, _ = initialize_psar_state(
            current_k_idx,
            high_arr,
            low_arr,
            close_arr,
            af0,
            -1,  # 强制空头
        )
        new_current_atr_tsl_price = (
            open_arr[current_k_idx]
            + atr_price_result[current_k_idx] * ATR_TSL_MULTIPLIER
        )

    # 平空进多 (4)
    elif (
        (
            trade_status_prev in nb_is_short_set
            or trade_status_prev in nb_is_no_position_set
        )
        and exit_short_prev
        and enter_long_prev
        and not enter_short_prev
        and not exit_long_prev
    ):
        new_trade_status = 4
        new_trigger_price = open_arr[current_k_idx]
        new_current_direction = 1
        new_psar_state, _ = initialize_psar_state(
            current_k_idx,
            high_arr,
            low_arr,
            close_arr,
            af0,
            1,  # 强制多头
        )
        new_current_atr_tsl_price = (
            open_arr[current_k_idx]
            - atr_price_result[current_k_idx] * ATR_TSL_MULTIPLIER
        )

    # 平多离场 (3)
    elif trade_status_prev in nb_is_long_set and exit_long_prev:
        new_trade_status = 3
        new_trigger_price = open_arr[current_k_idx]
        new_current_direction = 0
        new_psar_state = (False, np.nan, np.nan, np.nan)
        new_current_atr_tsl_price = np.nan

    # 平空离场 (-3)
    elif trade_status_prev in nb_is_short_set and exit_short_prev:
        new_trade_status = -3
        new_trigger_price = open_arr[current_k_idx]
        new_current_direction = 0
        new_psar_state = (False, np.nan, np.nan, np.nan)
        new_current_atr_tsl_price = np.nan

    # 多头进场 (1)
    elif (
        trade_status_prev in nb_is_no_position_set
        and enter_long_prev
        and not exit_long_prev
        and not enter_short_prev
    ):
        new_trade_status = 1
        new_trigger_price = open_arr[current_k_idx]
        new_current_direction = 1
        new_psar_state, _ = initialize_psar_state(
            current_k_idx,
            high_arr,
            low_arr,
            close_arr,
            af0,
            1,  # 强制多头
        )
        new_current_atr_tsl_price = (
            open_arr[current_k_idx]
            - atr_price_result[current_k_idx] * ATR_TSL_MULTIPLIER
        )

    # 空头进场 (-1)
    elif (
        trade_status_prev in nb_is_no_position_set
        and enter_short_prev
        and not exit_short_prev
        and not enter_long_prev
    ):
        new_trade_status = -1
        new_trigger_price = open_arr[current_k_idx]
        new_current_direction = -1
        new_psar_state, _ = initialize_psar_state(
            current_k_idx,
            high_arr,
            low_arr,
            close_arr,
            af0,
            -1,  # 强制空头
        )
        new_current_atr_tsl_price = (
            open_arr[current_k_idx]
            + atr_price_result[current_k_idx] * ATR_TSL_MULTIPLIER
        )

    # 如果没有新的开平仓信号，则保持默认状态（由 _update_initial_states_for_k_line 决定）
    else:
        new_trade_status = trade_status_prev  # 继承前一个K线的状态
        new_current_direction = current_direction_prev  # 继承前一个K线的方向
        # entry_price和current_atr_tsl_price也继承自prev，此函数不负责更新
        # psar_state 也不在此函数中更新，除非是初始化

    return (
        new_trade_status,
        new_trigger_price,
        new_current_direction,
        new_psar_state,
        new_current_atr_tsl_price,
    )
