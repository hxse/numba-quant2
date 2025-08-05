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
exit_signals_signature = nb.void(
    nb_int_type,  # i
    nb_int_type,  # last_i
    nb_float_type[:, :],  # tohlcv
    nb_float_type,  # target_price
    nb_bool_type[:, :],  # signal_result_child
    nb_float_type[:, :],  # backtest_result_child
    nb_bool_type[:],  # temp_psar_is_long
    nb_float_type[:],  # temp_psar_current
    nb_float_type[:],  # temp_psar_ep
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_LONG_POSITION
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_SHORT_POSITION
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_NO_POSITION
    nb_float_type,  # atr_sl_multiplier
    nb_float_type,  # atr_tp_multiplier
    nb_float_type,  # atr_tsl_multiplier
    nb_float_type,  # psar_af0
    nb_float_type,  # psar_af_step
    nb_float_type,  # psar_max_af
    nb_bool_type,  # close_for_reversal
)


@nb.jit(exit_signals_signature, nopython=True, cache=True)
def calculate_exit_triggers(
    i,
    last_i,
    tohlcv,
    target_price,
    signal_result_child,
    backtest_result_child,
    temp_psar_is_long,
    temp_psar_current,
    temp_psar_ep,
    IS_LONG_POSITION,
    IS_SHORT_POSITION,
    IS_NO_POSITION,
    atr_sl_multiplier,
    atr_tp_multiplier,
    atr_tsl_multiplier,
    psar_af0,
    psar_af_step,
    psar_max_af,
    close_for_reversal,
):
    """
    根据当前K线数据和仓位状态，计算止损止盈触发价格和离场信号。
    """

    # 6. 从 tohlcv 中提取时间、开盘、最高、最低、收盘、成交量数组
    time_arr = tohlcv[:, 0]
    open_arr = tohlcv[:, 1]
    high_arr = tohlcv[:, 2]
    low_arr = tohlcv[:, 3]
    close_arr = tohlcv[:, 4]
    volume_arr = tohlcv[:, 5]

    # 提取信号和回测结果数组
    enter_long_signal = signal_result_child[:, 0]
    exit_long_signal = signal_result_child[:, 1]
    enter_short_signal = signal_result_child[:, 2]
    exit_short_signal = signal_result_child[:, 3]

    # 提取回测结果数组
    position_status_result = backtest_result_child[:, 0]
    trigger_price_result = backtest_result_child[:, 1]
    atr_price_result = backtest_result_child[:, 2]
    atr_sl_price_result = backtest_result_child[:, 3]
    atr_tp_price_result = backtest_result_child[:, 4]
    atr_tsl_price_result = backtest_result_child[:, 5]
    psar_long_result = backtest_result_child[:, 6]
    psar_short_result = backtest_result_child[:, 7]
    psar_af_result = backtest_result_child[:, 8]
    psar_reversal_result = backtest_result_child[:, 9]

    atr = atr_price_result[i]
    atr_sl = atr * atr_sl_multiplier
    atr_tp = atr * atr_tp_multiplier
    atr_tsl = atr * atr_tsl_multiplier

    # 用i在当下仓位及时计算离场信号exit_long_trigger_result, 离场信号会在下一个循环触发交易
    # psar tsl和atr tsl行为类似,每次开仓都初始化,从而跟踪仓位状态
    if position_status_result[i] in (1, 4):  # 多头开仓或反手
        atr_sl_price_result[i] = trigger_price_result[i] - atr_sl
        atr_tp_price_result[i] = trigger_price_result[i] + atr_tp
        atr_tsl_price_result[i] = target_price - atr_tsl

        (
            temp_psar_is_long[i],
            temp_psar_current[i],
            temp_psar_ep[i],
            psar_af_result[i],
        ) = psar_init(
            high_arr[last_i],
            high_arr[i],
            low_arr[last_i],
            low_arr[i],
            close_arr[last_i],
            psar_af0,
            1,
        )
        (
            (
                temp_psar_is_long[i],
                temp_psar_current[i],
                temp_psar_ep[i],
                psar_af_result[i],
            ),
            psar_long_result[i],
            psar_short_result[i],
            psar_reversal_result[i],
        ) = psar_update(
            (
                temp_psar_is_long[i],
                temp_psar_current[i],
                temp_psar_ep[i],
                psar_af_result[i],
            ),
            high_arr[i],
            low_arr[i],
            high_arr[last_i],
            low_arr[last_i],
            psar_af_step,
            psar_max_af,
            close_arr[i],
            close_for_reversal,
        )

    elif position_status_result[i] in (-1, -4):  # 空头开仓或反手
        atr_sl_price_result[i] = trigger_price_result[i] + atr_sl
        atr_tp_price_result[i] = trigger_price_result[i] - atr_tp
        atr_tsl_price_result[i] = target_price + atr_tsl

        (
            temp_psar_is_long[i],
            temp_psar_current[i],
            temp_psar_ep[i],
            psar_af_result[i],
        ) = psar_init(
            high_arr[last_i],
            high_arr[i],
            low_arr[last_i],
            low_arr[i],
            close_arr[last_i],
            psar_af0,
            -1,
        )
        (
            (
                temp_psar_is_long[i],
                temp_psar_current[i],
                temp_psar_ep[i],
                psar_af_result[i],
            ),
            psar_long_result[i],
            psar_short_result[i],
            psar_reversal_result[i],
        ) = psar_update(
            (
                temp_psar_is_long[i],
                temp_psar_current[i],
                temp_psar_ep[i],
                psar_af_result[i],
            ),
            high_arr[i],
            low_arr[i],
            high_arr[last_i],
            low_arr[last_i],
            psar_af_step,
            psar_max_af,
            close_arr[i],
            close_for_reversal,
        )

    elif position_status_result[i] == 2:  # 多头持仓
        atr_sl_price_result[i] = atr_sl_price_result[last_i]
        atr_tp_price_result[i] = atr_tp_price_result[last_i]
        exit_price = close_arr[i] if close_for_reversal else high_arr[i]
        atr_tsl_price_result[i] = max(
            atr_tsl_price_result[last_i], exit_price - atr_tsl
        )

        (
            (
                temp_psar_is_long[i],
                temp_psar_current[i],
                temp_psar_ep[i],
                psar_af_result[i],
            ),
            psar_long_result[i],
            psar_short_result[i],
            psar_reversal_result[i],
        ) = psar_update(
            (
                temp_psar_is_long[last_i],
                temp_psar_current[last_i],
                temp_psar_ep[last_i],
                psar_af_result[last_i],
            ),
            high_arr[i],
            low_arr[i],
            high_arr[last_i],
            low_arr[last_i],
            psar_af_step,
            psar_max_af,
            close_arr[i],
            close_for_reversal,
        )

    elif position_status_result[i] == -2:  # 空头持仓
        atr_sl_price_result[i] = atr_sl_price_result[last_i]
        atr_tp_price_result[i] = atr_tp_price_result[last_i]
        exit_price = close_arr[i] if close_for_reversal else low_arr[i]
        atr_tsl_price_result[i] = min(
            atr_tsl_price_result[last_i], exit_price + atr_tsl
        )

        (
            (
                temp_psar_is_long[i],
                temp_psar_current[i],
                temp_psar_ep[i],
                psar_af_result[i],
            ),
            psar_long_result[i],
            psar_short_result[i],
            psar_reversal_result[i],
        ) = psar_update(
            (
                temp_psar_is_long[last_i],
                temp_psar_current[last_i],
                temp_psar_ep[last_i],
                psar_af_result[last_i],
            ),
            high_arr[i],
            low_arr[i],
            high_arr[last_i],
            low_arr[last_i],
            psar_af_step,
            psar_max_af,
            close_arr[i],
            close_for_reversal,
        )

    # 生成离场触发信号
    if position_status_result[i] in IS_LONG_POSITION:
        exit_price = close_arr[i] if close_for_reversal else low_arr[i]
        if (
            exit_price < atr_sl_price_result[i]
            or exit_price > atr_tp_price_result[i]
            or exit_price < atr_tsl_price_result[i]
        ):
            enter_long_signal[i] = False
            exit_long_signal[i] = True
            exit_short_signal[i] = True
        if psar_reversal_result[i] == 1:
            enter_long_signal[i] = False
            exit_long_signal[i] = True
            exit_short_signal[i] = True
    elif position_status_result[i] in IS_SHORT_POSITION:
        exit_price = close_arr[i] if close_for_reversal else high_arr[i]
        if (
            exit_price > atr_sl_price_result[i]
            or exit_price < atr_tp_price_result[i]
            or exit_price > atr_tsl_price_result[i]
        ):
            enter_long_signal[i] = False
            exit_long_signal[i] = True
            exit_short_signal[i] = True
        if psar_reversal_result[i] == 1:
            enter_long_signal[i] = False
            exit_long_signal[i] = True
            exit_short_signal[i] = True
