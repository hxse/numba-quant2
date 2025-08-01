import numba as nb
import numpy as np
from utils.data_types import indicator_params_child, indicator_result_child

from src.indicators.sma import sma_id, sma2_id, sma_name, sma2_name, sma_spec, sma2_spec
from src.indicators.bbands import bbands_id, bbands_name, bbands_spec

from .calculation_tool import (
    bool_compare,
    assign_elementwise,
    ComparisonOperator as co,
    AssignOperator as ao,
)


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


signature = nb.void(
    nb_float_type[:, :],  # tohlcv
    nb_float_type[:, :],  # tohlcv2
    indicator_result_child(
        nb_int_type, nb_float_type, nb_bool_type
    ),  # indicator_result_child
    indicator_result_child(
        nb_int_type, nb_float_type, nb_bool_type
    ),  # indicator_result2_child
    nb_int_type[:],  # signal_params
    nb_bool_type[:, :],  # signal_result_child
    nb.types.Tuple(
        (
            nb_int_type[:, :],  # int_temp_array_child
            nb_float_type[:, :],  # float_temp_array_child
            nb_bool_type[:, :],  # bool_temp_array_child
        )
    ),
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def simple_signal(
    tohlcv,
    tohlcv2,
    indicator_result_child,
    indicator_result2_child,
    signal_params,
    signal_result_child,
    temp_args,
):
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
    bool_compare(sma_result, sma2_result, enter_long_signal, co.gt, ao.ASSIGN)

    exit_long_signal = signal_result_child[:, 1]
    bool_compare(sma_result, sma2_result, exit_long_signal, co.lt, ao.ASSIGN)

    enter_short_signal = signal_result_child[:, 2]
    bool_compare(sma_result, sma2_result, enter_short_signal, co.lt, ao.ASSIGN)

    exit_short_signal = signal_result_child[:, 3]
    bool_compare(sma_result, sma2_result, exit_short_signal, co.gt, ao.ASSIGN)

    enter_long_signal = signal_result_child[:, 0]
    bool_compare(sma_result, sma2_result, enter_long_signal, co.gt, ao.BITWISE_AND)

    exit_long_signal = signal_result_child[:, 1]
    bool_compare(sma_result, sma2_result, exit_long_signal, co.lt, ao.BITWISE_AND)

    enter_short_signal = signal_result_child[:, 2]
    bool_compare(sma_result, sma2_result, enter_short_signal, co.lt, ao.BITWISE_AND)

    exit_short_signal = signal_result_child[:, 3]
    bool_compare(sma_result, sma2_result, exit_short_signal, co.gt, ao.BITWISE_AND)

    enter_long_signal = signal_result_child[:, 0]
    bool_compare(sma_result, sma2_result, enter_long_signal, co.gt, ao.BITWISE_OR)

    exit_long_signal = signal_result_child[:, 1]
    bool_compare(sma_result, sma2_result, exit_long_signal, co.lt, ao.BITWISE_OR)

    enter_short_signal = signal_result_child[:, 2]
    bool_compare(sma_result, sma2_result, enter_short_signal, co.lt, ao.BITWISE_OR)

    exit_short_signal = signal_result_child[:, 3]
    bool_compare(sma_result, sma2_result, exit_short_signal, co.gt, ao.BITWISE_OR)
