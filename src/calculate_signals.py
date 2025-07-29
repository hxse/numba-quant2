import numba as nb
import numpy as np
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types
from utils.data_types import default_types, get_params_child_signature

from src.indicators.sma import sma_id, sma2_id, sma_name, sma2_name, sma_spec, sma2_spec
from src.indicators.bbands import bbands_id, bbands_name, bbands_spec
from src.signal.simple_template import simple_signal


def calc_signal(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    nb_bool_type = dtype_dict["nb"]["bool"]

    params_child_signature = get_params_child_signature(
        nb_int_type, nb_float_type, nb_bool_type)
    signature = nb.void(params_child_signature)

    _simple_signal = simple_signal(mode, cache=cache, dtype_dict=dtype_dict)

    def _calc_signal(params_child):
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

        if len(signal_params) > 1:
            if signal_params[0] == 0 and signal_params[1] >= 0:
                _simple_signal(tohlcv, tohlcv2, indicator_result_child,
                               indicator_result2_child, signal_params,
                               signal_result_child, temp_args)

    return numba_wrapper(mode, signature=signature,
                         cache_enabled=cache)(_calc_signal)
