import numpy as np
import numba as nb
from utils.numba_utils import nb_wrapper
from utils.data_types import (
    get_params_signature,
    get_params_child_signature,
)
from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types


dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]

params_signature = get_params_signature(nb_int_type, nb_float_type, nb_bool_type)
signature = nb.void(params_signature)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def init_data(params):
    """
    不建议用这个,gpu模式下会一直卡住,用init_data_child吧
    """
    pass
    # (data_args, indicator_args, signal_args, backtest_args, temp_args) = params
    # (tohlcv, tohlcv2, tohlcv_smooth, tohlcv_smooth2, mapping_data) = data_args
    # (
    #     indicator_params,
    #     indicator_params2,
    #     indicator_enabled,
    #     indicator_enabled2,
    #     indicator_result,
    #     indicator_result2,
    # ) = indicator_args
    # (signal_params, signal_result) = signal_args
    # (backtest_params, backtest_result) = backtest_args
    # (
    #     int_temp_array,
    #     int_temp_array2,
    #     float_temp_array,
    #     float_temp_array2,
    #     bool_temp_array,
    #     bool_temp_array2,
    # ) = temp_args

    # (sma_result, sma2_result, bbands_result, atr_result, psar_result) = indicator_result
    # (sma_result2, sma2_result2, bbands_result2, atr_result2, psar_result2) = (
    #     indicator_result2
    # )

    # tohlcv_smooth[:] = np.nan
    # tohlcv_smooth2[:] = np.nan

    # for i in indicator_result:
    #     i[:] = np.nan
    # for i in indicator_result2:
    #     i[:] = np.nan

    # signal_result[:] = False
    # backtest_result[:] = np.nan
    # int_temp_array[:] = 0
    # int_temp_array2[:] = 0
    # float_temp_array[:] = np.nan
    # float_temp_array2[:] = np.nan
    # bool_temp_array[:] = False
    # bool_temp_array2[:] = False


params_signature = get_params_child_signature(nb_int_type, nb_float_type, nb_bool_type)
signature = nb.void(params_signature)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def init_data_child(params_child):
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

    tohlcv_smooth[:] = np.nan
    tohlcv_smooth2[:] = np.nan

    for i in indicator_result_child:
        i[:] = np.nan
    for i in indicator_result2_child:
        i[:] = np.nan

    signal_result_child[:] = False
    backtest_result_child[:] = np.nan
    int_temp_array_child[:] = 0
    int_temp_array2_child[:] = 0
    float_temp_array_child[:] = np.nan
    float_temp_array2_child[:] = np.nan
    bool_temp_array_child[:] = False
    bool_temp_array2_child[:] = False
