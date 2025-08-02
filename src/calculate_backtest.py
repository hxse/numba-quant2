import numba as nb
import numpy as np
from utils.data_types import get_params_child_signature


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper


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

    # 仓位触发价格,仅在1,-1,3,-3,4,-4时记录,0, 2,-2时不记录
    trigger_price_result = backtest_result_child[:, 2]

    # 初始化所有结果数组为0。这确保了在没有交易发生时，trigger_price_result 默认为0。
    trade_status_result[:] = 0
    current_direction_result[:] = 0
    trigger_price_result[:] = 0

    # 定义集合，用于归纳当前K线结束时的简化仓位方向
    IS_LONG_POSITION = (1, 2, 4)  # 多头进场, 持续多头, 平空开多
    IS_SHORT_POSITION = (-1, -2, -4)  # 空头进场, 持续空头, 平多开空
    IS_NO_POSITION = (0, 3, -3)  # 持续无仓位, 平多离场, 平空离场

    for i in range(1, len(time_arr)):  # 循环从第二根K线开始
        # 获取上一根K线的详细交易状态，用于当前分支判断
        trade_status_prev = trade_status_result[i - 1]

        # 获取上一根K线的进出场信号
        enter_long_prev = enter_long_signal[i - 1]
        enter_short_prev = enter_short_signal[i - 1]
        exit_long_prev = exit_long_signal[i - 1]
        exit_short_prev = exit_short_signal[i - 1]

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
        # 平多离场 (3)
        elif trade_status_prev in IS_LONG_POSITION and exit_long_prev:
            trade_status_result[i] = 3
            trigger_price_result[i] = open_arr[i]
        # 平空离场 (-3)
        elif trade_status_prev in IS_SHORT_POSITION and exit_short_prev:
            trade_status_result[i] = -3
            trigger_price_result[i] = open_arr[i]
        # 多头进场 (1)
        elif (
            trade_status_prev in IS_NO_POSITION
            and enter_long_prev
            and not exit_long_prev
            and not enter_short_prev
        ):
            trade_status_result[i] = 1
            trigger_price_result[i] = open_arr[i]
        # 空头进场 (-1)
        elif (
            trade_status_prev in IS_NO_POSITION
            and enter_short_prev
            and not exit_short_prev
            and not enter_long_prev
        ):
            trade_status_result[i] = -1
            trigger_price_result[i] = open_arr[i]
        # 持续多头 (2)
        elif trade_status_prev in IS_LONG_POSITION:
            trade_status_result[i] = 2
        # 持续空头 (-2)
        elif trade_status_prev in IS_SHORT_POSITION:
            trade_status_result[i] = -2

        # 在 trade_status_result[i] 确定后，归纳出 current_direction_result[i]
        current_status_i = trade_status_result[i]  # 获取当前K线最新的详细状态
        if current_status_i in IS_LONG_POSITION:
            current_direction_result[i] = 1  # 归纳为多头
        elif current_status_i in IS_SHORT_POSITION:
            current_direction_result[i] = -1  # 归纳为空头
        elif current_status_i in IS_NO_POSITION:
            current_direction_result[i] = 0  # 归纳为无仓位
