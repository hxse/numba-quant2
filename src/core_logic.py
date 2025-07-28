import numba as nb
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types, get_params_child_signature
from .calculate_indicators import calc_indicators
from .calculate_signals import calc_signal


def parallel_calc(mode, cache=True, dtype_dict=default_types):
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    params_child_signature = get_params_child_signature(
        nb_int_type, nb_float_type)
    signature = nb.void(params_child_signature)

    _calc_indicators = calc_indicators(mode,
                                       cache=cache,
                                       dtype_dict=dtype_dict)

    _calc_signal = calc_signal(mode, cache=cache, dtype_dict=dtype_dict)

    def _parallel_calc(params_child):
        _calc_indicators(params_child)
        _calc_signal(params_child)

    return numba_wrapper(mode, signature=signature,
                         cache_enabled=cache)(_parallel_calc)
