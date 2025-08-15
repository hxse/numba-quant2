import numba as nb
import numpy as np

from src.indicators.indicators_wrapper import IndicatorsId
from utils.data_types import (
    get_indicator_result_child,
    get_temp_result_child,
)


from .signal_tool import (
    bool_compare,
    ComparisonOperator as co,
    AssignOperator as ao,
    TriggerOperator as to,
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
    "dependency": {"sma": True, "sma2": True},
    "dependency2": {"sma": True, "sma2": True},
    "exit_control": {
        "pct_sl_enable": True,
        "pct_tp_enable": False,
        "pct_tsl_enable": False,
        "atr_sl_enable": True,
        "atr_tp_enable": False,
        "atr_tsl_enable": False,
        "psar_enable": False,
    },
}
simple_id = simple_spec["id"]
simple_name = simple_spec["name"]
simple_dependency = simple_spec["dependency"]
simple_dependency2 = simple_spec["dependency2"]

signature = nb.void(
    nb_float_type[:, :],  # tohlcv
    nb_float_type[:, :],  # tohlcv2
    get_indicator_result_child(nb_int_type, nb_float_type, nb_bool_type),
    get_indicator_result_child(nb_int_type, nb_float_type, nb_bool_type),
    nb_int_type[:],  # signal_params
    nb_bool_type[:, :],  # signal_result_child
    get_temp_result_child(nb_int_type, nb_float_type, nb_bool_type),
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
    (
        int_temp_array_child,
        int_temp_array2_child,
        float_temp_array_child,
        float_temp_array2_child,
        bool_temp_array_child,
        bool_temp_array2_child,
    ) = temp_args

    bool_temp_array = bool_temp_array_child[:, 0]

    close = tohlcv[:, 4]

    sma_indicator_result_child = indicator_result_child[IndicatorsId.sma.value]
    sma_result = sma_indicator_result_child[:, 0]

    sma2_indicator_result_child = indicator_result_child[IndicatorsId.sma2.value]
    sma2_result = sma2_indicator_result_child[:, 0]

    sma_indicator_result2_child = indicator_result2_child[IndicatorsId.sma.value]
    sma_result2 = sma_indicator_result2_child[:, 0]

    sma2_indicator_result2_child = indicator_result2_child[IndicatorsId.sma2.value]
    sma2_result2 = sma2_indicator_result2_child[:, 0]

    enter_long_signal = signal_result_child[:, 0]
    exit_long_signal = signal_result_child[:, 1]
    enter_short_signal = signal_result_child[:, 2]
    exit_short_signal = signal_result_child[:, 3]

    bool_compare(
        sma_result,
        sma2_result,
        enter_long_signal,
        bool_temp_array,
        co.gt,
        ao.ASSIGN,
        to.EDGE,
    )

    bool_compare(
        sma_result,
        sma2_result,
        exit_long_signal,
        bool_temp_array,
        co.lt,
        ao.ASSIGN,
        to.CONTINUOUS,
    )

    bool_compare(
        sma_result,
        sma2_result,
        enter_short_signal,
        bool_temp_array,
        co.lt,
        ao.ASSIGN,
        to.CONTINUOUS,
    )

    bool_compare(
        sma_result,
        sma2_result,
        exit_short_signal,
        bool_temp_array,
        co.gt,
        ao.ASSIGN,
        to.CONTINUOUS,
    )

    bool_compare(
        sma_result,
        sma2_result,
        enter_long_signal,
        bool_temp_array,
        co.gt,
        ao.BITWISE_AND,
        to.CONTINUOUS,
    )

    bool_compare(
        sma_result,
        sma2_result,
        exit_long_signal,
        bool_temp_array,
        co.lt,
        ao.BITWISE_AND,
        to.CONTINUOUS,
    )

    bool_compare(
        sma_result,
        sma2_result,
        enter_short_signal,
        bool_temp_array,
        co.lt,
        ao.BITWISE_AND,
        to.CONTINUOUS,
    )

    bool_compare(
        sma_result,
        sma2_result,
        exit_short_signal,
        bool_temp_array,
        co.gt,
        ao.BITWISE_AND,
        to.CONTINUOUS,
    )

    bool_compare(
        sma_result,
        sma2_result,
        enter_long_signal,
        bool_temp_array,
        co.gt,
        ao.BITWISE_OR,
        to.CONTINUOUS,
    )

    bool_compare(
        sma_result,
        sma2_result,
        exit_long_signal,
        bool_temp_array,
        co.lt,
        ao.BITWISE_OR,
        to.CONTINUOUS,
    )

    bool_compare(
        sma_result,
        sma2_result,
        enter_short_signal,
        bool_temp_array,
        co.lt,
        ao.BITWISE_OR,
        to.CONTINUOUS,
    )

    bool_compare(
        sma_result,
        sma2_result,
        exit_short_signal,
        bool_temp_array,
        co.gt,
        ao.BITWISE_OR,
        to.CONTINUOUS,
    )
