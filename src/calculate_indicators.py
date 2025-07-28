import numba as nb
import numpy as np
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types, get_params_child_signature
from src.indicators.sma import calculate_sma_wrapper, sma_id, sma2_id
from src.indicators.bbands import calculate_bbands_wrapper, bbands_id


def calc_indicators(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    params_child_signature = get_params_child_signature(
        nb_int_type, nb_float_type)
    signature = nb.void(params_child_signature)

    _calculate_sma_wrapper = calculate_sma_wrapper(mode,
                                                   cache=cache,
                                                   dtype_dict=dtype_dict)

    _calculate_bbands_wrapper = calculate_bbands_wrapper(mode,
                                                         cache=cache,
                                                         dtype_dict=dtype_dict)

    def _calc_indicators(params_child):
        print("INSIDE _calc_indicators: Function started.")

        (data_args, indicator_args, signal_args, backtest_args) = params_child
        (tohlcv, tohlcv2, tohlcv_smooth, tohlcv_smooth2) = data_args
        (indicator_params_child, indicator_params2_child, indicator_enabled,
         indicator_enabled2, indicator_result_child,
         indicator_result2_child) = indicator_args
        (signal_params, signal_result_child) = signal_args
        (backtest_params_child, backtest_result_child,
         temp_arrays_child) = backtest_args

        (sma_params, sma2_params, bbands_params) = indicator_params_child
        (sma_params2, sma2_params2, bbands_params2) = indicator_params2_child
        (sma_result, sma2_result, bbands_result) = indicator_result_child
        (sma_result2, sma2_result2, bbands_result2) = indicator_result2_child
        if indicator_enabled[sma_id] == 1:

            _calculate_sma_wrapper(tohlcv, indicator_params_child,
                                   indicator_result_child, sma_id)

        if indicator_enabled[sma2_id] == 1:

            _calculate_sma_wrapper(tohlcv, indicator_params_child,
                                   indicator_result_child, sma2_id)

        if indicator_enabled[bbands_id] == 1:

            _calculate_bbands_wrapper(tohlcv, indicator_params_child,
                                      indicator_result_child, bbands_id)

    return numba_wrapper(mode, signature=signature,
                         cache_enabled=cache)(_calc_indicators)
