import numba as nb
import numpy as np
import math
from utils.data_types import get_params_child_signature


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper


dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


signature = nb.void(
    nb_int_type,  # i
    nb_int_type,  # last_i
    nb_float_type,  # target_price
    nb_bool_type[:, :],  # signal_result_child
    nb_float_type[:, :],  # backtest_result_child
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_LONG_POSITION
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_SHORT_POSITION
    nb.types.Tuple((nb_int_type, nb_int_type, nb_int_type)),  # IS_NO_POSITION
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def process_trade_logic(
    i,
    last_i,
    target_price,
    signal_result_child,
    backtest_result_child,
    IS_LONG_POSITION,
    IS_SHORT_POSITION,
    IS_NO_POSITION,
):
    """
    处理交易逻辑：根据前一根K线的信号和当前仓位状态，更新当前K线的仓位状态和触发价格。
    """
    # 提取信号和回测结果数组
    enter_long_signal = signal_result_child[:, 0]
    exit_long_signal = signal_result_child[:, 1]
    enter_short_signal = signal_result_child[:, 2]
    exit_short_signal = signal_result_child[:, 3]

    position_status_result = backtest_result_child[:, 0]
    entry_price_result = backtest_result_child[:, 1]
    exit_price_result = backtest_result_child[:, 2]

    # 根据上一根K线仓位，更新当前仓位为“持有”
    if position_status_result[last_i] in IS_LONG_POSITION:
        position_status_result[i] = 2
        entry_price_result[i] = entry_price_result[last_i]
    elif position_status_result[last_i] in IS_SHORT_POSITION:
        position_status_result[i] = -2
        entry_price_result[i] = entry_price_result[last_i]

    # 根据信号处理开平仓逻辑
    if (
        enter_long_signal[last_i]
        and exit_short_signal[last_i]
        and position_status_result[i] in IS_SHORT_POSITION
    ):
        position_status_result[i] = 4  # 反手
        entry_price_result[i] = target_price
        exit_price_result[i] = target_price
    elif (
        enter_short_signal[last_i]
        and exit_long_signal[last_i]
        and position_status_result[i] in IS_LONG_POSITION
    ):
        position_status_result[i] = -4  # 反手
        entry_price_result[i] = target_price
        exit_price_result[i] = target_price
    elif exit_long_signal[last_i] and position_status_result[i] in IS_LONG_POSITION:
        position_status_result[i] = 3  # 平仓
        exit_price_result[i] = target_price
    elif exit_short_signal[last_i] and position_status_result[i] in IS_SHORT_POSITION:
        position_status_result[i] = -3  # 平仓
        exit_price_result[i] = target_price
    elif enter_long_signal[last_i] and position_status_result[i] in IS_NO_POSITION:
        position_status_result[i] = 1  # 开多
        entry_price_result[i] = target_price
    elif enter_short_signal[last_i] and position_status_result[i] in IS_NO_POSITION:
        position_status_result[i] = -1  # 开空
        entry_price_result[i] = target_price
