import numba as nb
import numpy as np
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types, get_params_child_signature
from src.indicators.sma import calculate_sma_wrapper, sma_id, sma2_id
from src.indicators.bbands import calculate_bbands_wrapper, bbands_id


def calc_indicators(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    nb_bool_type = dtype_dict["nb"]["bool"]

    params_child_signature = get_params_child_signature(
        nb_int_type, nb_float_type, nb_bool_type)
    signature = nb.void(params_child_signature)

    _calculate_sma_wrapper = calculate_sma_wrapper(mode,
                                                   cache=cache,
                                                   dtype_dict=dtype_dict)

    _calculate_bbands_wrapper = calculate_bbands_wrapper(mode,
                                                         cache=cache,
                                                         dtype_dict=dtype_dict)

    def _calc_indicators(params_child):
        (data_args, indicator_args, signal_args, backtest_args,
         temp_args) = params_child
        (tohlcv, tohlcv2, tohlcv_smooth, tohlcv_smooth2,
         mapping_data) = data_args
        (indicator_params_child, indicator_params2_child, indicator_enabled,
         indicator_enabled2, indicator_result_child,
         indicator_result2_child) = indicator_args
        (signal_params, signal_result_child) = signal_args
        (backtest_params_child, backtest_result_child) = backtest_args
        (int_temp_array_child, float_temp_array_child,
         bool_temp_array_child) = temp_args

        if indicator_enabled[sma_id]:
            _calculate_sma_wrapper(tohlcv, indicator_params_child,
                                   indicator_result_child, sma_id)

        if indicator_enabled[sma2_id]:
            _calculate_sma_wrapper(tohlcv, indicator_params_child,
                                   indicator_result_child, sma2_id)

        if indicator_enabled[bbands_id]:
            _calculate_bbands_wrapper(tohlcv, indicator_params_child,
                                      indicator_result_child, bbands_id)

    return numba_wrapper(mode, signature=signature,
                         cache_enabled=cache)(_calc_indicators)
