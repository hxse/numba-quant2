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


from src.backtest.psar_stop_loss import apply_psar_stop_loss

# 导入 ATR exit type 和相关函数
from src.backtest.atr_exit_logic import AtrExitType
from src.backtest.atr_exit_logic import apply_atr_exit_logic


# --- 辅助函数签名 ---
signature = nb.types.Tuple(
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
    nb_float_type[:],  # high_arr
    nb_float_type[:],  # low_arr
    nb_float_type[:],  # close_arr
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


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def apply_exit_strategies(  # 函数名略微调整
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
            # high_arr,
            # low_arr,
            # close_arr,
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
