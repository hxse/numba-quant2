import numba as nb
import numpy as np
import math
from utils.data_types import get_params_child_signature


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper


from src.indicators.psar import psar_init, psar_update
from src.backtest.psar_stop_loss import apply_psar_stop_loss
from src.backtest.psar_initialization import initialize_psar_state

from src.indicators.atr import calculate_atr
from src.backtest.atr_exit_logic import apply_atr_exit_logic, AtrExitType


# 导入新封装的辅助函数
from src.backtest.update_states import update_initial_states_for_k_line
from src.backtest.apply_exit_strategies import apply_exit_strategies
from src.backtest.handle_signals import handle_entry_exit_signals


dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


params_child_signature = get_params_child_signature(
    nb_int_type, nb_float_type, nb_bool_type
)
signature = nb.void(params_child_signature)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calc_backtest(params_child):
    (data_args, indicator_args, signal_args, backtest_args, temp_args) = params_child
    (tohlcv, tohlcv2, tohlcv_smooth, tohlcv_smooth2, mapping_data) = data_args
    (
        indicator_params_child,
        indicator_params2_child,
        indicator_enabled,
        indicator_enabled2,
        indicator_result_child,
        indicator_result2_child,
    ) = indicator_args
    (signal_params, signal_result_child) = signal_args
    (backtest_params_child, backtest_result_child) = backtest_args
    (int_temp_array_child, float_temp_array_child, bool_temp_array_child) = temp_args

    time_arr = tohlcv[:, 0]
    open_arr = tohlcv[:, 1]
    high_arr = tohlcv[:, 2]
    low_arr = tohlcv[:, 3]
    close_arr = tohlcv[:, 4]
    volume_arr = tohlcv[:, 5]

    enter_long_signal = signal_result_child[:, 0]
    exit_long_signal = signal_result_child[:, 1]
    enter_short_signal = signal_result_child[:, 2]
    exit_short_signal = signal_result_child[:, 3]

    # position_result: 0无仓位,1多头,2持多,3平多,4平空开多,-1空头,-2持空,-3平空,-4平多开空
    trade_status_result = backtest_result_child[:, 0]

    # current_position: 0无仓位,1多头,-1空头
    current_direction_result = backtest_result_child[:, 1]

    # 仓位触发价格,仅在1,-1,3,-3,4,-4时记录,0,2,-2时不记录
    trigger_price_result = backtest_result_child[:, 2]

    # 记录 PSAR 跟踪止损的价格 (用于可视化)
    psar_price_result = backtest_result_child[:, 3]

    # 记录 atr 的全局指标
    atr_price_result = backtest_result_child[:, 4]

    # 记录 atr sl 的数据
    atr_sl_price_result = backtest_result_child[:, 5]

    # 记录 atr tp 的数据
    atr_tp_price_result = backtest_result_child[:, 6]

    # 记录 atr tsl 的数据
    atr_tsl_price_result = backtest_result_child[:, 7]

    # 初始化所有结果数组为 0 或 NaN
    trade_status_result[:] = 0
    current_direction_result[:] = 0
    trigger_price_result[:] = np.nan
    psar_price_result[:] = np.nan
    atr_sl_price_result[:] = np.nan
    atr_tp_price_result[:] = np.nan
    atr_tsl_price_result[:] = np.nan

    # 定义集合，用于归纳当前 K 线结束时的简化仓位方向 (作为数组传递给 Numba 函数)
    IS_LONG_POSITION = (1, 2, 4)
    IS_SHORT_POSITION = (-1, -2, -4)
    IS_NO_POSITION = (0, 3, -3)

    # PSAR 参数（假设从 indicator_params_child 获取）
    # 实际项目中，这些参数应该从 indicator_params_child 获取
    psar_params = indicator_params_child[4]  # 假设 psar_id = 4
    af0 = psar_params[0]  # 初始加速因子
    af_step = psar_params[1]  # 加速因子步长
    max_af = psar_params[2]  # 最大加速因子

    # ATR 参数（硬编码，以后可从 indicator_params_child 获取）
    ATR_PERIOD = 14
    ATR_SL_MULTIPLIER = 2.0  # 止损乘数，例如 2 倍 ATR
    ATR_TP_MULTIPLIER = 3.0  # 止盈乘数，例如 3 倍 ATR
    ATR_TSL_MULTIPLIER = 2.0  # 跟踪止损乘数，例如 2 倍 ATR

    # --- 全局计算 ATR 指标 ---
    # 获取一个临时的浮点数组用于 calculate_atr 的 tr_result 参数
    temp_tr_array = float_temp_array_child[0]
    temp_tr_array[:] = np.nan  # 确保临时数组被清空

    # 调用 calculate_atr 计算整个历史数据范围内的 ATR 值
    calculate_atr(
        high_arr, low_arr, close_arr, ATR_PERIOD, atr_price_result, temp_tr_array
    )

    # PSAR 中间状态变量
    psar_state = (False, np.nan, np.nan, np.nan)

    # ATR TSL 跟踪止损价格（随 K 线更新）
    current_atr_tsl_price = np.nan
    # 入场价格 (用于 ATR SL/TP/TSL 的基准)
    entry_price = np.nan

    for i in range(1, len(time_arr)):  # 循环从第二根 K 线开始 (i=1)
        # 获取上一根 K 线的详细交易状态和方向
        trade_status_prev = trade_status_result[i - 1]
        current_direction_prev = current_direction_result[i - 1]

        # --- 1. 更新当前 K 线的初始状态 ---
        # 继承上一 K 线状态，并根据是否有新开仓更新 entry_price 和 current_atr_tsl_price
        (
            entry_price,
            current_atr_tsl_price,
            default_current_direction,
            default_trade_status,
        ) = update_initial_states_for_k_line(
            i,
            trade_status_prev,
            current_direction_prev,
            open_arr,
            atr_price_result,
            ATR_TSL_MULTIPLIER,
            entry_price,
            current_atr_tsl_price,
        )
        trade_status_result[i] = default_trade_status
        current_direction_result[i] = default_current_direction

        # --- 2. 应用所有离场策略（止损/止盈） ---
        (
            triggered_by_exit,
            new_trade_status_after_exit,
            new_trigger_price_after_exit,
            new_current_direction_after_exit,
            new_psar_state_after_exit,
            new_atr_tsl_price_after_exit,
        ) = apply_exit_strategies(
            psar_state,
            current_direction_prev,
            high_arr,
            low_arr,
            close_arr,
            af0,
            af_step,
            max_af,
            i,
            close_arr[i],
            atr_price_result[i],
            entry_price,
            ATR_SL_MULTIPLIER,
            ATR_TP_MULTIPLIER,
            ATR_TSL_MULTIPLIER,
            current_atr_tsl_price,
        )

        # 更新全局状态变量
        psar_state = new_psar_state_after_exit
        current_atr_tsl_price = new_atr_tsl_price_after_exit

        # 记录 ATR SL/TP 值 (用于可视化)
        if current_direction_result[i] == 1 and not math.isnan(
            atr_price_result[i]
        ):  # 多头
            atr_sl_price_result[i] = (
                entry_price - atr_price_result[i] * ATR_SL_MULTIPLIER
            )
            atr_tp_price_result[i] = (
                entry_price + atr_price_result[i] * ATR_TP_MULTIPLIER
            )
        elif current_direction_result[i] == -1 and not math.isnan(
            atr_price_result[i]
        ):  # 空头
            atr_sl_price_result[i] = (
                entry_price + atr_price_result[i] * ATR_SL_MULTIPLIER
            )
            atr_tp_price_result[i] = (
                entry_price - atr_price_result[i] * ATR_TP_MULTIPLIER
            )
        else:
            atr_sl_price_result[i] = np.nan
            atr_tp_price_result[i] = np.nan

        # 更新 PSAR display price (它可能在 apply_psar_stop_loss 内部设置，这里仅确保其在每次循环中都正确更新)
        # PSAR display price 应该由 apply_psar_stop_loss 返回，并在这里直接使用
        _, _, _, psar_display_price_val = (
            apply_psar_stop_loss(  # 再次调用以获取最新的显示价格，这有点重复，但为了模块化可接受
                psar_state,
                current_direction_result[i],  # 使用当前的direction，因为可能已反转
                high_arr[i],
                low_arr[i],
                high_arr[i - 1],
                low_arr[i - 1],
                af0,
                af_step,
                max_af,
                i,
                # high_arr,
                # low_arr,
                # close_arr,
            )
        )
        psar_price_result[i] = psar_display_price_val
        atr_tsl_price_result[i] = current_atr_tsl_price  # 记录当前的ATR TSL价格

        # 如果被任何离场逻辑触发，则更新状态并跳过后续的开平仓信号处理
        if triggered_by_exit:
            trade_status_result[i] = new_trade_status_after_exit
            trigger_price_result[i] = new_trigger_price_after_exit
            current_direction_result[i] = new_current_direction_after_exit
            entry_price = np.nan  # 离场后入场价清空
            # psar_state 和 current_atr_tsl_price 已在 _apply_exit_strategies 中处理
            continue  # 跳到下一根K线

        # --- 3. 处理开平仓信号 (如果未被任何离场逻辑触发) ---
        enter_long_prev = signal_result_child[i - 1, 0]
        exit_long_prev = signal_result_child[i - 1, 1]
        enter_short_prev = signal_result_child[i - 1, 2]
        exit_short_prev = signal_result_child[i - 1, 3]

        (
            new_trade_status,
            new_trigger_price,
            new_current_direction,
            new_psar_state_after_signals,
            new_atr_tsl_price_after_signals,
        ) = handle_entry_exit_signals(
            i,
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
        )

        # 更新全局状态变量
        trade_status_result[i] = new_trade_status
        trigger_price_result[i] = new_trigger_price
        current_direction_result[i] = new_current_direction
        psar_state = new_psar_state_after_signals
        current_atr_tsl_price = new_atr_tsl_price_after_signals

        # 如果是新的开仓，更新 entry_price
        if (
            new_trade_status == 1
            or new_trade_status == -1
            or new_trade_status == 4
            or new_trade_status == -4
        ):
            entry_price = new_trigger_price  # 开仓价格
        elif new_trade_status == 3 or new_trade_status == -3:  # 平仓
            entry_price = np.nan  # 平仓后清空

        # 更新 PSAR display price 和 ATR TSL price (确保所有路径都更新)
        # PSAR display price 应该由 apply_psar_stop_loss 返回，并在这里直接使用
        _, _, _, psar_display_price_val = (
            apply_psar_stop_loss(  # 再次调用以获取最新的显示价格
                psar_state,
                current_direction_result[i],  # 使用当前的direction
                high_arr[i],
                low_arr[i],
                high_arr[i - 1],
                low_arr[i - 1],
                af0,
                af_step,
                max_af,
                i,
                # high_arr,
                # low_arr,
                # close_arr,
            )
        )
        psar_price_result[i] = psar_display_price_val
        atr_tsl_price_result[i] = current_atr_tsl_price  # 记录当前的ATR TSL价格
