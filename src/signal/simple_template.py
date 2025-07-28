import numba as nb
import numpy as np
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types

from src.indicators.sma import sma_id, sma2_id, sma_name, sma2_name, sma_spec, sma2_spec
from src.indicators.bbands import bbands_id, bbands_name, bbands_spec


def simple_signal(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    signature = nb.void(
        nb_float_type[:, :],  # tohlcv
        nb.types.Tuple((nb_float_type[:, :], nb_float_type[:, :],
                        nb_float_type[:, :])),  # indicator_result_child
        nb.types.Tuple((nb_float_type[:, :], nb_float_type[:, :],
                        nb_float_type[:, :])),  # indicator_result2_child
        nb_int_type[:],  # signal_params
        nb_float_type[:, :]  # signal_result_child
    )

    def _simple_signal(tohlcv, indicator_result_child, indicator_result2_child,
                       signal_params, signal_result_child):

        close = tohlcv[:, 4]

        sma_indicator_result_child = indicator_result_child[sma_id]
        sma_result = sma_indicator_result_child[:, 0]

        sma2_indicator_result_child = indicator_result_child[sma2_id]
        sma2_result = sma2_indicator_result_child[:, 0]

        sma_indicator_result2_child = indicator_result2_child[sma_id]
        sma_result2 = sma_indicator_result2_child[:, 0]

        sma2_indicator_result2_child = indicator_result2_child[sma2_id]
        sma2_result2 = sma2_indicator_result2_child[:, 0]

        enter_long_signal = signal_result_child[:, 0]
        for i in range(len(enter_long_signal)):
            enter_long_signal[i] = int(sma_result[i] > sma2_result[i])

        exit_long_signal = signal_result_child[:, 1]
        for i in range(len(exit_long_signal)):
            exit_long_signal[i] = int(sma_result[i] < sma2_result[i])

        enter_short_signal = signal_result_child[:, 2]
        for i in range(len(enter_short_signal)):
            enter_short_signal[i] = int(sma_result[i] < sma2_result[i])

        exit_short_signal = signal_result_child[:, 3]
        for i in range(len(exit_short_signal)):
            exit_short_signal[i] = int(sma_result[i] > sma2_result[i])

    return numba_wrapper(mode, signature=signature,
                         cache_enabled=cache)(_simple_signal)
