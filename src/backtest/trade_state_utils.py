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

# 导入 ATR exit type
from src.backtest_utils.atr_stop_loss import AtrExitType
from src.backtest_utils.psar_stop_loss import (
    apply_psar_stop_loss,
)  # 确保已从backtest_utils导入
from src.backtest_utils.atr_stop_loss import (
    apply_atr_exit_logic,
)  # 确保已从backtest_utils导入
from src.backtest_utils.psar_initialization import (
    initialize_psar_state,
)  # 确保已从backtest_utils导入


# --- 辅助函数签名 ---
# _update_initial_states_for_k_line 的签名
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
    nb.types.Array(nb_float_type, 1, "C", readonly=True),  # open_arr
    nb.types.Array(nb_float_type, 1, "C", readonly=True),  # atr_price_result
    nb_float_type,  # ATR_TSL_MULTIPLIER
    nb_float_type,  # prev_entry_price
    nb_float_type,  # prev_current_atr_tsl_price
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def _update_initial_states_for_k_line(
    current_k_idx,
    trade_status_prev,
    current_direction_prev,
    open_arr,
    atr_price_result,
    ATR_TSL_MULTIPLIER,
    prev_entry_price,  # 传入上一根K线的entry_price
    prev_current_atr_tsl_price,  # 传入上一根K线的current_atr_tsl_price
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


# _apply_exit_strategies 的签名
signature_apply_exit_strategies = nb.types.Tuple(
    (
        nb_bool_type,  # was_triggered (是否被任何离场逻辑触发)
        nb_int_type,  # new_trade_status
        nb_float_type,  # new_trigger_price
        nb_int_type,  # new_current_direction
        nb.types.Tuple(
            (nb_bool_type, nb_float_type, nb_float_type, nb_float_type)
        ),  # new_psar_state
        nb_float_type,  # updated_atr_tsl_price
    )
)(
    # PSAR inputs
    nb.types.Tuple(
        (nb_bool_type, nb_float_type, nb_float_type, nb_float_type)
    ),  # psar_state
    nb_int_type,  # current_direction_prev
    nb.types.Array(nb_float_type, 1, "C", readonly=True),  # high_arr
    nb.types.Array(nb_float_type, 1, "C", readonly=True),  # low_arr
    nb.types.Array(nb_float_type, 1, "C", readonly=True),  # close_arr
    nb_float_type,  # af0
    nb_float_type,  # af_step
    nb_float_type,  # max_af
    nb_int_type,  # current_k_idx
    # ATR inputs
    nb_float_type,  # current_close
    nb_float_type,  # current_atr_value
    nb_float_type,  # entry_price
    nb_float_type,  # ATR_SL_MULTIPLIER
    nb_float_type,  # ATR_TP_MULTIPLIER
    nb_float_type,  # ATR_TSL_MULTIPLIER
    nb_float_type,  # current_atr_tsl_price
)


@nb.jit(nopython=True, cache=nb_params.get("cache", True), fastmath=True)
def _apply_exit_strategies(
    psar_state,
    current_direction_prev,
    high_arr,
    low_arr,
    close_arr,
    af0,
    af_step,
    max_af,
    current_k_idx,
    current_close,
    current_atr_value,
    entry_price,
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    ATR_TSL_MULTIPLIER,
    current_atr_tsl_price,
):
    """
    统一处理所有离场策略（止损、止盈、跟踪止损）的判断和优先级。
    返回触发状态和更新后的相关变量。
    """
    triggered = False
    new_trade_status = 0
    new_trigger_price = np.nan
    new_current_direction = 0
    new_psar_state = psar_state
    updated_atr_tsl_price = current_atr_tsl_price  # 默认不更新

    # --- 1. 应用 PSAR 跟踪止损逻辑 ---
    triggered_by_psar, psar_trigger_price, temp_psar_state, psar_display_price_val = (
        apply_psar_stop_loss(
            psar_state,
            current_direction_prev,
            high_arr[current_k_idx],
            low_arr[current_k_idx],
            high_arr[current_k_idx - 1],  # 注意这里的索引
            low_arr[current_k_idx - 1],  # 注意这里的索引
            af0,
            af_step,
            max_af,
            current_k_idx,
            high_arr,
            low_arr,
            close_arr,
        )
    )
    # 临时更新 psar_state，如果 PSAR 触发了最终离场，会被清空
    new_psar_state = temp_psar_state

    # --- 2. 应用 ATR 止损/止盈/跟踪止损逻辑 ---
    # 止损 (SL)
    triggered_by_atr_sl, atr_sl_trigger_price, _ = apply_atr_exit_logic(
        AtrExitType.SL,
        current_direction_prev,
        current_close,
        current_atr_value,
        entry_price,
        ATR_SL_MULTIPLIER,
        np.nan,
    )
    # 止盈 (TP)
    triggered_by_atr_tp, atr_tp_trigger_price, _ = apply_atr_exit_logic(
        AtrExitType.TP,
        current_direction_prev,
        current_close,
        current_atr_value,
        entry_price,
        ATR_TP_MULTIPLIER,
        np.nan,
    )
    # 跟踪止损 (TSL)
    triggered_by_atr_tsl, atr_tsl_trigger_price, temp_updated_tsl_price = (
        apply_atr_exit_logic(
            AtrExitType.TSL,
            current_direction_prev,
            current_close,
            current_atr_value,
            entry_price,
            ATR_TSL_MULTIPLIER,
            current_atr_tsl_price,
        )
    )
    # 临时更新 TSL 价格，如果其他止损触发最终离场，会被清空
    updated_atr_tsl_price = temp_updated_tsl_price

    # --- 3. 优先级判断和状态更新 ---
    # 止损优先级最高，如果任何一种止损触发，立即平仓
    if triggered_by_psar:
        triggered = True
        new_trade_status = 3 if current_direction_prev == 1 else -3
        new_trigger_price = psar_trigger_price
        new_current_direction = 0
        new_psar_state = (False, np.nan, np.nan, np.nan)  # 止损后 PSAR 状态清空
        updated_atr_tsl_price = np.nan  # 止损后 ATR TSL 清空
    elif triggered_by_atr_sl:
        triggered = True
        new_trade_status = 3 if current_direction_prev == 1 else -3
        new_trigger_price = atr_sl_trigger_price
        new_current_direction = 0
        new_psar_state = (False, np.nan, np.nan, np.nan)  # 止损后 PSAR 状态清空
        updated_atr_tsl_price = np.nan  # 止损后 ATR TSL 清空
    elif triggered_by_atr_tsl:
        triggered = True
        new_trade_status = 3 if current_direction_prev == 1 else -3
        new_trigger_price = atr_tsl_trigger_price
        new_current_direction = 0
        new_psar_state = (False, np.nan, np.nan, np.nan)  # 止损后 PSAR 状态清空
        updated_atr_tsl_price = np.nan  # 止损后 ATR TSL 清空
    elif triggered_by_atr_tp:  # 止盈在所有止损之后
        triggered = True
        new_trade_status = 3 if current_direction_prev == 1 else -3
        new_trigger_price = atr_tp_trigger_price
        new_current_direction = 0
        new_psar_state = (False, np.nan, np.nan, np.nan)  # 止盈后 PSAR 状态清空
        updated_atr_tsl_price = np.nan  # 止盈后 ATR TSL 清空

    return (
        triggered,
        new_trade_status,
        new_trigger_price,
        new_current_direction,
        new_psar_state,
        updated_atr_tsl_price,
    )


# _handle_entry_exit_signals 的签名
signature_handle_signals = nb.types.Tuple(
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
    nb.types.Array(nb_float_type, 1, "C", readonly=True),  # open_arr
    nb.types.Array(nb_float_type, 1, "C", readonly=True),  # high_arr
    nb.types.Array(nb_float_type, 1, "C", readonly=True),  # low_arr
    nb.types.Array(nb_float_type, 1, "C", readonly=True),  # close_arr
    nb.types.Array(nb_float_type, 1, "C", readonly=True),  # atr_price_result
    nb_float_type,  # af0
    nb_float_type,  # ATR_TSL_MULTIPLIER
    nb.types.Array(nb_int_type, 1, "C", readonly=True),  # IS_LONG_POSITION
    nb.types.Array(nb_int_type, 1, "C", readonly=True),  # IS_SHORT_POSITION
    nb.types.Array(nb_int_type, 1, "C", readonly=True),  # IS_NO_POSITION
)


@nb.jit(nopython=True, cache=nb_params.get("cache", True), fastmath=True)
def _handle_entry_exit_signals(
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
