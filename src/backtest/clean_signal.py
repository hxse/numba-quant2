import numba as nb
import numpy as np
from utils.numba_utils import nb_wrapper
from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from src.indicators.atr import calculate_atr

# 定义 Numba 数据类型
dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_float_type = dtype_dict["nb"]["float"]
nb_int_type = dtype_dict["nb"]["int"]
nb_bool_type = dtype_dict["nb"]["bool"]


from utils.data_types import get_params_child_signature


signal = nb.void(
    nb_bool_type[:, :],  # signal_result_child
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signal,
    cache_enabled=nb_params.get("cache", True),
)
def clean_signal(signal_result_child):
    enter_long_signal = signal_result_child[:, 0]
    exit_long_signal = signal_result_child[:, 1]
    enter_short_signal = signal_result_child[:, 2]
    exit_short_signal = signal_result_child[:, 3]

    for i in range(signal_result_child.shape[0]):
        if enter_long_signal[i] and exit_long_signal[i]:
            enter_long_signal[i] = False
            # exit_long_signal[i]=False
        if enter_short_signal[i] and exit_short_signal[i]:
            enter_short_signal[i] = False
            # exit_short_signal[i]=False
        if enter_long_signal[i] and enter_short_signal[i]:
            enter_long_signal[i] = False
            enter_short_signal[i] = False
