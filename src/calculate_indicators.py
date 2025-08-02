import numba as nb
import numpy as np
from utils.data_types import get_params_child_signature
from src.indicators.sma import calculate_sma_wrapper, sma_id, sma2_id
from src.indicators.bbands import calculate_bbands_wrapper, bbands_id
from src.indicators.atr import calculate_atr_wrapper, atr_id
from src.indicators.psar import calculate_psar_wrapper, psar_id

from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


params_child_signature = get_params_child_signature(
    nb_int_type, nb_float_type, nb_bool_type
)
signature = nb.void(params_child_signature)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calc_indicators(params_child):
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
    (int_temp_array_child, float_temp_array_child, bool_temp_array_child) = temp_args

    id_arr = (sma_id, sma2_id, bbands_id, atr_id, psar_id)
    func_arr = (
        calculate_sma_wrapper,
        calculate_sma_wrapper,
        calculate_bbands_wrapper,
        calculate_atr_wrapper,
        calculate_psar_wrapper,
    )

    for i in range(len(id_arr)):
        func_arr[i](
            tohlcv,
            indicator_params_child,
            indicator_result_child,
            float_temp_array_child,
            id_arr[i],
        )
