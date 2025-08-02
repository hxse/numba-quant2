import numba as nb
import numpy as np

from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


signature = nb_int_type(
    nb_float_type[:],
    nb_int_type,
    nb_float_type[:],
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def check_bounds(close, period, result):
    data_length = len(close)
    result_length = len(result)

    # 越界检查
    if result_length < data_length:
        return 0

    # 越界检查
    if period <= 0 or data_length < period or result_length < period:
        return 0

    return 1
