import numba as nb
import numpy as np


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper


from utils.data_types import loop_indicators_signature


from enum import IntEnum, auto


from .sma import (
    calculate_sma,
    calculate_sma_wrapper,
    sma_spec,
    sma2_spec,
)
from .bbands import calculate_bbands, calculate_bbands_wrapper, bbands_spec
from .atr import calculate_atr, calculate_atr_wrapper, atr_spec
from .psar import calculate_psar, calculate_psar_wrapper, psar_spec


dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


class IndicatorsId(IntEnum):
    sma = 0
    sma2 = auto()
    bbands = auto()
    atr = auto()
    psar = auto()


indicators_id_array = tuple(member.value for member in IndicatorsId)


indicators_spec = {
    "sma": {
        "id": IndicatorsId.sma,
        **sma_spec,
        "func": calculate_sma,
        "func_wrapper": calculate_sma_wrapper,
    },
    "sma2": {
        "id": IndicatorsId.sma2,
        **sma2_spec,
        "func": calculate_sma,
        "func_wrapper": calculate_sma_wrapper,
    },
    "bbands": {
        "id": IndicatorsId.bbands,  # 指标id，不能跟其他指标重复。
        **bbands_spec,
        "func": calculate_bbands,
        "func_wrapper": calculate_bbands_wrapper,
    },
    "atr": {
        "id": IndicatorsId.atr,
        **atr_spec,
        "func": calculate_atr,
        "func_wrapper": calculate_atr_wrapper,
    },
    "psar": {
        "id": IndicatorsId.psar,
        **psar_spec,
        "func": calculate_psar,
        "func_wrapper": calculate_psar_wrapper,
    },
}


indicators_func_arr = tuple(v["func_wrapper"] for k, v in indicators_spec.items())


signature = nb.void(
    *loop_indicators_signature(nb_int_type, nb_float_type, nb_bool_type)
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def loop_indicators(
    indicator_id, tohlcv, indicator_params, indicator_result, float_temp_array
):
    if indicator_id == IndicatorsId.sma:
        calculate_sma_wrapper(
            indicator_id, tohlcv, indicator_params, indicator_result, float_temp_array
        )
    elif indicator_id == IndicatorsId.sma2:
        calculate_sma_wrapper(
            indicator_id, tohlcv, indicator_params, indicator_result, float_temp_array
        )
    elif indicator_id == IndicatorsId.bbands:
        calculate_bbands_wrapper(
            indicator_id, tohlcv, indicator_params, indicator_result, float_temp_array
        )
    elif indicator_id == IndicatorsId.atr:
        calculate_atr_wrapper(
            indicator_id, tohlcv, indicator_params, indicator_result, float_temp_array
        )
    elif indicator_id == IndicatorsId.psar:
        calculate_psar_wrapper(
            indicator_id, tohlcv, indicator_params, indicator_result, float_temp_array
        )
