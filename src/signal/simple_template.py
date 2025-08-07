import numba as nb
import numpy as np
from utils.data_types import get_indicator_params_child, get_indicator_result_child

from src.indicators.sma import sma_id, sma2_id, sma_name, sma2_name, sma_spec, sma2_spec
from src.indicators.bbands import bbands_id, bbands_name, bbands_spec

from .signal_tool import (
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


simple_spec = {
    "id": 0,
    "name": "simple",
    "dependency": {sma_name: True, sma2_name: True},
    "dependency2": {sma_name: True, sma2_name: True},
}
simple_id = simple_spec["id"]
simple_name = simple_spec["name"]
simple_dependency = simple_spec["dependency"]
simple_dependency2 = simple_spec["dependency2"]

signature = nb.void(
    nb_float_type[:, :],  # tohlcv
    nb_float_type[:, :],  # tohlcv2
    get_indicator_result_child(
        nb_int_type, nb_float_type, nb_bool_type
    ),  # indicator_result_child
    get_indicator_result_child(
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
    exit_long_signal = signal_result_child[:, 1]
    enter_short_signal = signal_result_child[:, 2]
    exit_short_signal = signal_result_child[:, 3]

    bool_compare(sma_result, sma2_result, enter_long_signal, co.gt, ao.ASSIGN)

    bool_compare(sma_result, sma2_result, exit_long_signal, co.lt, ao.ASSIGN)

    bool_compare(sma_result, sma2_result, enter_short_signal, co.lt, ao.ASSIGN)

    bool_compare(sma_result, sma2_result, exit_short_signal, co.gt, ao.ASSIGN)

    bool_compare(sma_result, sma2_result, enter_long_signal, co.gt, ao.BITWISE_AND)

    bool_compare(sma_result, sma2_result, exit_long_signal, co.lt, ao.BITWISE_AND)

    bool_compare(sma_result, sma2_result, enter_short_signal, co.lt, ao.BITWISE_AND)

    bool_compare(sma_result, sma2_result, exit_short_signal, co.gt, ao.BITWISE_AND)

    bool_compare(sma_result, sma2_result, enter_long_signal, co.gt, ao.BITWISE_OR)

    bool_compare(sma_result, sma2_result, exit_long_signal, co.lt, ao.BITWISE_OR)

    bool_compare(sma_result, sma2_result, enter_short_signal, co.lt, ao.BITWISE_OR)

    bool_compare(sma_result, sma2_result, exit_short_signal, co.gt, ao.BITWISE_OR)
