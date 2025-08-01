import numpy as np
import numba as nb
from utils.numba_utils import numba_wrapper, nb_wrapper
from utils.data_types import (
    default_types,
    get_params_signature,
    get_params_child_signature,
)
from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types

from src.indicators.sma import sma_spec, sma2_spec
from src.indicators.bbands import bbands_spec


def initialize_outputs(
    tohlcv,
    tohlcv2,
    indicator_params,
    indicator_params2,
    indicator_enabled,
    indicator_enabled2,
    conf_count,
    dtype_dict,
    temp_int_num,
    temp_float_num,
    temp_bool_num,
    min_rows=0,
):
    """
    初始化并返回所有计算所需的输出数组和临时数组。

    参数:
    - tohlcv_shape: 形状元组，表示主OHLCV数据的形状 (rows, cols)。
    - tohlcv2_shape: 形状元组，表示第二组OHLCV数据的形状 (rows, cols)。
    - conf_count: 并发策略的数量 (对应于 backtest_params 的第一维)。
    - dtype_dict: 包含 numpy 和 numba 数据类型的字典。
    - temp_num: 临时数组的最后一维大小。

    返回:
    一个元组，包含所有初始化好的 numpy 数组：
    (tohlcv_smooth, tohlcv_smooth2,
     indicator_result, indicator_result2,
     signal_result, signal_result2,
     backtest_result, temp_arrays)
    """

    # todo 等稳定了, 再让AI简化一下这个函数,现在先不弄

    np_int_type = dtype_dict["np"]["int"]
    np_float_type = dtype_dict["np"]["float"]
    np_bool_type = dtype_dict["np"]["bool"]

    tohlcv_shape = tohlcv.shape
    tohlcv2_shape = tohlcv2.shape
    tohlcv_rows = tohlcv_shape[0]
    tohlcv2_rows = tohlcv2_shape[0]

    # --- Data Smooth Arrays ---
    tohlcv_smooth = np.full(tohlcv_shape, np.nan, dtype=np_float_type)
    tohlcv_smooth2 = np.full(tohlcv2_shape, np.nan, dtype=np_float_type)

    # --- Indicator Result Arrays ---
    sma_output_dim = sma_spec["result_count"]
    sma2_output_dim = sma2_spec["result_count"]
    bbands_output_dim = bbands_spec["result_count"]

    signal_output_dim = 4
    backtest_output_dim = 10
    temp_float_num = max(
        temp_float_num,
        sma_spec["temp_count"] + sma2_spec["temp_count"] + bbands_spec["temp_count"],
    )

    sma_rows = tohlcv_rows if indicator_enabled[sma_spec["id"]] else min_rows
    sma_result = np.full(
        (conf_count, sma_rows, sma_output_dim), np.nan, dtype=np_float_type
    )

    sma2_rows = tohlcv_rows if indicator_enabled[sma2_spec["id"]] else min_rows
    sma2_result = np.full(
        (conf_count, sma2_rows, sma2_output_dim), np.nan, dtype=np_float_type
    )

    bbands_rows = tohlcv_rows if indicator_enabled[bbands_spec["id"]] else min_rows
    bbands_result = np.full(
        (conf_count, bbands_rows, bbands_output_dim), np.nan, dtype=np_float_type
    )

    indicator_result = (sma_result, sma2_result, bbands_result)

    sma_rows2 = tohlcv2_rows if indicator_enabled2[sma_spec["id"]] else min_rows
    sma_result2 = np.full(
        (conf_count, sma_rows2, sma_output_dim), np.nan, dtype=np_float_type
    )

    sma2_rows2 = tohlcv2_rows if indicator_enabled2[sma2_spec["id"]] else min_rows
    sma2_result2 = np.full(
        (conf_count, sma2_rows2, sma_output_dim), np.nan, dtype=np_float_type
    )

    bbands_rows2 = tohlcv2_rows if indicator_enabled2[bbands_spec["id"]] else min_rows
    bbands_result2 = np.full(
        (conf_count, bbands_rows2, bbands_output_dim), np.nan, dtype=np_float_type
    )
    indicator_result2 = (sma_result2, sma2_result2, bbands_result2)

    # --- Signal Result Arrays ---
    signal_result = np.full(
        (conf_count, tohlcv_rows, signal_output_dim), False, dtype=np_bool_type
    )

    # --- Backtest Result Array ---
    backtest_result = np.full(
        (conf_count, tohlcv_rows, backtest_output_dim), np.nan, dtype=np_float_type
    )

    # --- Temporary Arrays ---
    int_temp_array = np.full(
        (conf_count, tohlcv_rows, temp_int_num), 0, dtype=np_int_type
    )
    float_temp_array = np.full(
        (conf_count, tohlcv_rows, temp_float_num), 0, dtype=np_float_type
    )
    bool_temp_array = np.full(
        (conf_count, tohlcv_rows, temp_bool_num), 0, dtype=np_bool_type
    )

    return (
        tohlcv_smooth,
        tohlcv_smooth2,
        indicator_result,
        indicator_result2,
        signal_result,
        backtest_result,
        int_temp_array,
        float_temp_array,
        bool_temp_array,
    )


def unpack_params(
    outputs,
    tohlcv,
    tohlcv2,
    mapping_data,
    indicator_params,
    indicator_params2,
    indicator_enabled,
    indicator_enabled2,
    signal_params,
    backtest_params,
):
    (
        tohlcv_smooth,
        tohlcv_smooth2,
        indicator_result,
        indicator_result2,
        signal_result,
        backtest_result,
        int_temp_array,
        float_temp_array,
        bool_temp_array,
    ) = outputs

    data_args = (tohlcv, tohlcv2, tohlcv_smooth, tohlcv_smooth2, mapping_data)
    indicator_args = (
        indicator_params,
        indicator_params2,
        indicator_enabled,
        indicator_enabled2,
        indicator_result,
        indicator_result2,
    )
    signal_args = (signal_params, signal_result)
    backtest_args = (backtest_params, backtest_result)
    temp_args = (int_temp_array, float_temp_array, bool_temp_array)

    cpu_params = (data_args, indicator_args, signal_args, backtest_args, temp_args)
    return cpu_params


dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]

params_type = get_params_signature(nb_int_type, nb_float_type, nb_bool_type)
return_type = get_params_child_signature(nb_int_type, nb_float_type, nb_bool_type)
signature = return_type(params_type, nb_int_type)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def unpack_params_child(params, idx):
    """
    indicator_enabled, indicator_enabled2, signal_params 不需要用idx传递
    data_args, 不需要用idx传递
    其他都需要用idx传递
    """
    (data_args, indicator_args, signal_args, backtest_args, temp_args) = params
    (tohlcv, tohlcv2, tohlcv_smooth, tohlcv_smooth2, mapping_data) = data_args
    (
        indicator_params,
        indicator_params2,
        indicator_enabled,
        indicator_enabled2,
        indicator_result,
        indicator_result2,
    ) = indicator_args
    (signal_params, signal_result) = signal_args
    (backtest_params, backtest_result) = backtest_args
    (int_temp_array, float_temp_array, bool_temp_array) = temp_args

    (sma_params, sma2_params, bbands_params) = indicator_params
    (sma_params2, sma2_params2, bbands_params2) = indicator_params2
    (sma_result, sma2_result, bbands_result) = indicator_result
    (sma_result2, sma2_result2, bbands_result2) = indicator_result2

    indicator_params_child = (sma_params[idx], sma2_params[idx], bbands_params[idx])
    indicator_params2_child = (
        sma_params2[idx],
        sma2_params2[idx],
        bbands_params2[idx],
    )
    indicator_result_child = (sma_result[idx], sma2_result[idx], bbands_result[idx])
    indicator_result2_child = (
        sma_result2[idx],
        sma2_result2[idx],
        bbands_result2[idx],
    )

    indicator_args_child = (
        indicator_params_child,
        indicator_params2_child,
        indicator_enabled,
        indicator_enabled2,
        indicator_result_child,
        indicator_result2_child,
    )

    signal_args_child = (signal_params, signal_result[idx])

    backtest_args_child = (backtest_params[idx], backtest_result[idx])

    temp_args_child = (
        int_temp_array[idx],
        float_temp_array[idx],
        bool_temp_array[idx],
    )

    params_child = (
        data_args,
        indicator_args_child,
        signal_args_child,
        backtest_args_child,
        temp_args_child,
    )

    return params_child


def get_output(params):
    (data_args, indicator_args, signal_args, backtest_args, temp_args) = params
    (
        indicator_params,
        indicator_params2,
        indicator_enabled,
        indicator_enabled2,
        indicator_result,
        indicator_result2,
    ) = indicator_args
    (signal_params, signal_result) = signal_args
    (backtest_params, backtest_result) = backtest_args
    (int_temp_array, float_temp_array, bool_temp_array) = temp_args

    return (
        indicator_result,
        indicator_result2,
        signal_result,
        backtest_result,
        int_temp_array,
        float_temp_array,
        bool_temp_array,
    )


dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]

params_signature = get_params_signature(nb_int_type, nb_float_type, nb_bool_type)
signature = nb_int_type(params_signature)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def get_conf_count(params):
    (data_args, indicator_args, signal_args, backtest_args, temp_args) = params
    (backtest_params, backtest_result) = backtest_args
    return backtest_params.shape[0]
