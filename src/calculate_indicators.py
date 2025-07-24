import numba as nb
import numpy as np
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types, get_signature_child
from src.indicators.sma import calculate_sma_wrapper
from src.indicators.bbands import calculate_bbands_wrapper

_dict = {"sma_period": 0}


def calc_indicators(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    signature = get_signature_child(nb_int_type, nb_float_type)

    _calculate_sma_wrapper = calculate_sma_wrapper(mode,
                                                   cache=cache,
                                                   dtype_dict=dtype_dict)

    _calculate_bbands_wrapper = calculate_bbands_wrapper(mode,
                                                         cache=cache,
                                                         dtype_dict=dtype_dict)

    def _calc_indicators(params_child):
        (micro_data, micro_input_child, macro_data, macro_input_child,
         backtest_input_child) = params_child

        (micro_tohlcv, micro_tohlcv_smooth) = micro_data
        (micro_indicator_params_child, micro_signal_params_child,
         micro_indicator_result_child,
         micro_signal_result_child) = micro_input_child
        (macro_tohlcv, macro_tohlcv_smooth) = macro_data
        (macro_indicator_params_child, macro_signal_params_child,
         macro_indicator_result_child,
         macro_signal_result_child) = macro_input_child
        (backtest_params_child, backtest_result_child,
         temp_arrays_child) = backtest_input_child

        _calculate_sma_wrapper(micro_tohlcv, micro_indicator_params_child,
                               micro_indicator_result_child)
        result = micro_indicator_result_child[:, 0]

        _calculate_bbands_wrapper(micro_tohlcv, micro_indicator_params_child,
                                  micro_indicator_result_child)

    return numba_wrapper(mode, signature=signature,
                         cache_enabled=cache)(_calc_indicators)
