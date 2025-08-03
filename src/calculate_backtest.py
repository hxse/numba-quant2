import numba as nb
import numpy as np
import math
from utils.data_types import get_params_child_signature


from src.indicators.psar import psar_init, psar_update


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

from src.backtest.psar_stop_loss import apply_psar_stop_loss
from src.backtest.psar_initialization import initialize_psar_state


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

    # 初始化所有结果数组为 0 或 NaN
    trade_status_result[:] = 0
    current_direction_result[:] = 0
    trigger_price_result[:] = np.nan
    psar_price_result[:] = np.nan

    # 定义集合，用于归纳当前 K 线结束时的简化仓位方向
    IS_LONG_POSITION = (1, 2, 4)  # 多头进场, 持续多头, 平空开多
    IS_SHORT_POSITION = (-1, -2, -4)  # 空头进场, 持续空头, 平多开空
    IS_NO_POSITION = (0, 3, -3)  # 持续无仓位, 平多离场, 平空离场

    # PSAR 参数（假设从 indicator_params_child 获取）
    psar_params = indicator_params_child[4]  # 假设 psar_id = 4
    af0 = psar_params[0]  # 初始加速因子
    af_step = psar_params[1]  # 加速因子步长
    max_af = psar_params[2]  # 最大加速因子

    # PSAR 中间状态变量
    # 显式初始化为包含 NaN 的元组
    psar_state = (False, np.nan, np.nan, np.nan)

    for i in range(1, len(time_arr)):  # 循环从第二根 K 线开始 (i=1)
        # 获取上一根 K 线的详细交易状态和方向
        trade_status_prev = trade_status_result[i - 1]
        current_direction_prev = current_direction_result[i - 1]

        # 获取上一根 K 线（即 i-1 周期）的进出场信号
        enter_long_prev = enter_long_signal[i - 1]
        enter_short_prev = enter_short_signal[i - 1]
        exit_long_prev = exit_long_signal[i - 1]
        exit_short_prev = exit_short_signal[i - 1]

        # --- 1. 应用 PSAR 跟踪止损逻辑 ---
        triggered_by_psar, trigger_price_val, new_psar_state, psar_display_price_val = (
            apply_psar_stop_loss(
                psar_state,
                current_direction_prev,
                high_arr[i],
                low_arr[i],
                high_arr[i - 1],
                low_arr[i - 1],
                af0,
                af_step,
                max_af,
                i,  # 当前K线的索引用于psar_init的绝对位置
                high_arr,  # 完整的高点数组
                low_arr,  # 完整的低点数组
                close_arr,  # 完整的收盘价数组
            )
        )

        # 更新 PSAR 状态和显示价格
        psar_state = new_psar_state
        psar_price_result[i] = psar_display_price_val

        # 如果被 PSAR 止损触发，则处理并跳过后续的开平仓信号处理
        if triggered_by_psar:
            trade_status_result[i] = (
                3 if current_direction_prev == 1 else -3
            )  # 平多或平空
            trigger_price_result[i] = trigger_price_val
            current_direction_result[i] = 0  # 止损后方向为0
            continue  # 跳到下一根K线

        # --- 2. 处理其他交易信号（如果未被 PSAR 止损触发） ---
        # 默认情况下，继承上一K线的状态，如果无仓位则为0
        if trade_status_prev in IS_LONG_POSITION:
            trade_status_result[i] = 2
            current_direction_result[i] = 1
        elif trade_status_prev in IS_SHORT_POSITION:
            trade_status_result[i] = -2
            current_direction_result[i] = -1
        else:
            trade_status_result[i] = 0
            current_direction_result[i] = 0
            # 无仓位时 psar_price_result[i] 已经由 apply_psar_stop_loss 设置为 NaN
            # psar_state 也由 apply_psar_stop_loss 确保为无效

        # 优先处理开平仓反转 (平多开空或平空开多)
        # 平多进空 (-4)
        if (
            (
                trade_status_prev in IS_LONG_POSITION
                or trade_status_prev in IS_NO_POSITION
            )
            and exit_long_prev
            and enter_short_prev
            and not enter_long_prev
            and not exit_short_prev
        ):
            trade_status_result[i] = -4
            trigger_price_result[i] = open_arr[i]
            current_direction_result[i] = -1
            # 封装后的 PSAR 初始化
            psar_state, psar_price_result[i] = initialize_psar_state(
                i,
                high_arr,
                low_arr,
                close_arr,
                af0,
                -1,  # 强制空头
            )

        # 平空进多 (4)
        elif (
            (
                trade_status_prev in IS_SHORT_POSITION
                or trade_status_prev in IS_NO_POSITION
            )
            and exit_short_prev
            and enter_long_prev
            and not enter_short_prev
            and not exit_long_prev
        ):
            trade_status_result[i] = 4
            trigger_price_result[i] = open_arr[i]
            current_direction_result[i] = 1
            # 封装后的 PSAR 初始化
            psar_state, psar_price_result[i] = initialize_psar_state(
                i,
                high_arr,
                low_arr,
                close_arr,
                af0,
                1,  # 强制多头
            )

        # 平多离场 (3)
        elif trade_status_prev in IS_LONG_POSITION and exit_long_prev:
            trade_status_result[i] = 3
            trigger_price_result[i] = open_arr[i]
            current_direction_result[i] = 0
            psar_price_result[i] = np.nan  # 离场后PSAR清空
            # 平仓后，将 psar_state 重置为无效值
            psar_state = (False, np.nan, np.nan, np.nan)

        # 平空离场 (-3)
        elif trade_status_prev in IS_SHORT_POSITION and exit_short_prev:
            trade_status_result[i] = -3
            trigger_price_result[i] = open_arr[i]
            current_direction_result[i] = 0
            psar_price_result[i] = np.nan  # 离场后PSAR清空
            # 平仓后，将 psar_state 重置为无效值
            psar_state = (False, np.nan, np.nan, np.nan)

        # 多头进场 (1)
        elif (
            trade_status_prev in IS_NO_POSITION
            and enter_long_prev
            and not exit_long_prev
            and not enter_short_prev
        ):
            trade_status_result[i] = 1
            trigger_price_result[i] = open_arr[i]
            current_direction_result[i] = 1
            # 封装后的 PSAR 初始化
            psar_state, psar_price_result[i] = initialize_psar_state(
                i,
                high_arr,
                low_arr,
                close_arr,
                af0,
                1,  # 强制多头
            )

        # 空头进场 (-1)
        elif (
            trade_status_prev in IS_NO_POSITION
            and enter_short_prev
            and not exit_short_prev
            and not enter_long_prev
        ):
            trade_status_result[i] = -1
            trigger_price_result[i] = open_arr[i]
            current_direction_result[i] = -1
            # 封装后的 PSAR 初始化
            psar_state, psar_price_result[i] = initialize_psar_state(
                i,
                high_arr,
                low_arr,
                close_arr,
                af0,
                -1,  # 强制空头
            )
