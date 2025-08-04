import numba as nb
import numpy as np
import math
from utils.data_types import get_params_child_signature


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper


from src.indicators.psar import psar_init, psar_update

from src.indicators.atr import calculate_atr


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
    # 1. 解包 params_child 中的数据数组
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

    temp_tr_array = float_temp_array_child[:, 0]  # temp_tr_array

    temp_psar_is_long = float_temp_array_child[:, 1]  # temp_psar_is_long
    temp_psar_current = float_temp_array_child[:, 2]  # temp_psar_current
    temp_psar_ep = float_temp_array_child[:, 3]  # temp_psar_ep

    # 6. 从 tohlcv 中提取时间、开盘、最高、最低、收盘、成交量数组
    time_arr = tohlcv[:, 0]
    open_arr = tohlcv[:, 1]
    high_arr = tohlcv[:, 2]
    low_arr = tohlcv[:, 3]
    close_arr = tohlcv[:, 4]
    volume_arr = tohlcv[:, 5]

    # 7. 从 signal_result_child 中提取信号数组
    enter_long_signal = signal_result_child[:, 0]
    exit_long_signal = signal_result_child[:, 1]
    enter_short_signal = signal_result_child[:, 2]
    exit_short_signal = signal_result_child[:, 3]

    # 从 backtest_result_child 中提取回测结果数组
    position_status_result = backtest_result_child[:, 0]
    trigger_price_result = backtest_result_child[:, 1]
    exit_long_trigger_result = backtest_result_child[:, 2]
    exit_short_trigger_result = backtest_result_child[:, 3]
    # atr 相关
    atr_price_result = backtest_result_child[:, 4]
    atr_sl_price_result = backtest_result_child[:, 5]
    atr_tp_price_result = backtest_result_child[:, 6]
    atr_tsl_price_result = backtest_result_child[:, 7]
    # psar 相关
    psar_long_result = backtest_result_child[:, 8]
    psar_short_result = backtest_result_child[:, 9]
    psar_af_result = backtest_result_child[:, 10]
    psar_reversal_result = backtest_result_child[:, 11]
    # 还有, temp_psar_is_long, temp_psar_current, temp_psar_ep

    # 0无仓位,1开多,2持多,3平多,4平空开多,-1开空,-2持空,-3平空,-4平多开空
    position_status_result[:] = 0
    trigger_price_result[:] = np.nan
    exit_long_trigger_result[:] = 0  # 0,1 是否触发long离场
    exit_short_trigger_result[:] = 0  # 0,1 是否触发short离场
    atr_price_result[:] = np.nan
    atr_sl_price_result[:] = np.nan
    atr_tp_price_result[:] = np.nan
    atr_tsl_price_result[:] = np.nan

    # 定义 IS_LONG_POSITION, IS_SHORT_POSITION, IS_NO_POSITION 集合
    IS_LONG_POSITION = (1, 2, 4)
    IS_SHORT_POSITION = (-1, -2, -4)
    IS_NO_POSITION = (0, 3, -3)

    # PSAR参数, 目前先硬编码
    af0 = 0.02
    af_step = 0.02
    max_af = 0.2

    # ATR参数, 目前先硬编码
    atr_preiod = 14
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 2.0
    atr_tsl_multiplier = 2.0

    slippage_multiplier = 0.5  # 0.5倍atr的滑点

    # 全局计算atr
    calculate_atr(
        high_arr,
        low_arr,
        close_arr,
        atr_preiod,
        atr_price_result,
        temp_tr_array,
    )

    for i in range(1, len(time_arr)):  # 循环从第二根 K 线开始 (i=1)
        # 尽量不要用中间变量, 直接使用array[i]或array[i-1]来访问, 中间变量会让代码变的难以维护

        # 用last_i 来获取是一个信号,是为了确保在进场离场信号生成后的下一根k线的开盘价进行交易
        last_i = i - 1
        target_price = open_arr[i]

        if position_status_result[last_i] in IS_LONG_POSITION:
            position_status_result[i] = 2
            trigger_price_result[i] = trigger_price_result[i - 1]
        elif position_status_result[last_i] in IS_SHORT_POSITION:
            position_status_result[i] = -2
            trigger_price_result[i] = trigger_price_result[i - 1]

        if (
            enter_long_signal[last_i]
            and exit_short_signal[last_i]
            and position_status_result[i] in IS_SHORT_POSITION
        ):
            position_status_result[i] = 4
            trigger_price_result[i] = target_price  # 反手忽略滑点
        elif (
            enter_short_signal[last_i]
            and exit_long_signal[last_i]
            and position_status_result[i] in IS_LONG_POSITION
        ):
            position_status_result[i] = -4
            trigger_price_result[i] = target_price  # 反手忽略滑点
        elif (
            exit_long_signal[last_i]
            or exit_long_trigger_result[last_i]
            and position_status_result[i] in IS_LONG_POSITION
        ):
            position_status_result[i] = 3
            # slippage_price = target_price - atr_price_result[i] * slippage_multiplier
            # trigger_price_result[i] = slippage_price
            trigger_price_result[i] = target_price  # 建议后续计算资金再计算滑点
        elif (
            exit_short_signal[last_i]
            or exit_short_trigger_result[last_i]
            and position_status_result[i] in IS_SHORT_POSITION
        ):
            position_status_result[i] = -3
            # slippage_price = target_price + atr_price_result[i] * slippage_multiplier
            # trigger_price_result[i] = slippage_price
            trigger_price_result[i] = target_price  # 建议后续计算资金再计算滑点
        elif enter_long_signal[last_i] and position_status_result[i] in IS_NO_POSITION:
            position_status_result[i] = 1
            # slippage_price = target_price + atr_price_result[i] * slippage_multiplier
            # trigger_price_result[i] = slippage_price
            trigger_price_result[i] = target_price  # 建议后续计算资金再计算滑点
        elif enter_short_signal[last_i] and position_status_result[i] in IS_NO_POSITION:
            position_status_result[i] = -1
            # slippage_price = target_price - atr_price_result[i] * slippage_multiplier
            # trigger_price_result[i] = slippage_price
            trigger_price_result[i] = target_price  # 建议后续计算资金再计算滑点

        atr = atr_price_result[i]
        atr_sl = atr * atr_sl_multiplier
        atr_tp = atr * atr_tp_multiplier
        atr_tsl = atr * atr_tsl_multiplier

        # 用i在当下仓位及时计算离场信号exit_long_trigger_result, 离场信号会在下一个循环触发交易
        exit_price = close_arr[i]

        if position_status_result[i] in (1, 4):
            atr_sl_price_result[i] = trigger_price_result[i] - atr_sl
            atr_tp_price_result[i] = trigger_price_result[i] + atr_tp
            atr_tsl_price_result[i] = exit_price - atr_tsl
        elif position_status_result[i] in (-1, -4):
            atr_sl_price_result[i] = trigger_price_result[i] + atr_sl
            atr_tp_price_result[i] = trigger_price_result[i] - atr_tp
            atr_tsl_price_result[i] = exit_price + atr_tsl
        elif position_status_result[i] == 2:
            atr_sl_price_result[i] = atr_sl_price_result[i - 1]
            atr_tp_price_result[i] = atr_tp_price_result[i - 1]
            atr_tsl_price_result[i] = max(
                atr_tsl_price_result[i - 1], exit_price - atr_tsl
            )
        elif position_status_result[i] == -2:
            atr_sl_price_result[i] = atr_sl_price_result[i - 1]
            atr_tp_price_result[i] = atr_tp_price_result[i - 1]
            atr_tsl_price_result[i] = min(
                atr_tsl_price_result[i - 1], exit_price + atr_tsl
            )

        if position_status_result[i] in IS_LONG_POSITION:
            if (
                exit_price < atr_sl_price_result[i]
                or exit_price > atr_tp_price_result[i]
                or exit_price < atr_tsl_price_result[i]
            ):
                exit_long_trigger_result[i] = 1
        elif position_status_result[i] in IS_SHORT_POSITION:
            if (
                exit_price > atr_sl_price_result[i]
                or exit_price < atr_tp_price_result[i]
                or exit_price > atr_tsl_price_result[i]
            ):
                exit_short_trigger_result[i] = 1
