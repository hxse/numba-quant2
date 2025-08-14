import numba as nb
import numpy as np
from utils.data_types import get_params_child_signature

from src.backtest.clean_signal import clean_signal
from src.signal.simple_template import simple_signal, simple_id


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


signal_result_name = [
    "enter_long",
    "exit_long",
    "enter_short",
    "exit_short",
]
signal_result_count = len(signal_result_name)


params_child_signature = get_params_child_signature(
    nb_int_type, nb_float_type, nb_bool_type
)
signature = nb.void(params_child_signature)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calc_signal(params_child):
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
    (
        int_temp_array_child,
        int_temp_array2_child,
        float_temp_array_child,
        float_temp_array2_child,
        bool_temp_array_child,
        bool_temp_array2_child,
    ) = temp_args

    id_array = (simple_id,)
    func_array = (simple_signal,)

    for i in range(len(id_array)):
        if len(signal_params) > 0:
            if signal_params[0] == id_array[i]:
                func_array[i](
                    tohlcv,
                    tohlcv2,
                    indicator_result_child,
                    indicator_result2_child,
                    signal_params,
                    signal_result_child,
                    temp_args,
                )

    clean_signal(signal_result_child)
