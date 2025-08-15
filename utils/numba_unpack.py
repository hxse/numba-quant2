import numpy as np
import numba as nb
from utils.numba_utils import nb_wrapper
from utils.data_types import (
    get_params_signature,
    get_params_child_signature,
    get_numba_data_types,
)
from src.indicators.indicators_wrapper import indicators_spec
from src.calculate_signals import signal_result_count
from src.backtest.calculate_backtest import backtest_result_count


from utils.numba_params import nb_params

dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


def create_array(
    mode: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    fill: bool = False,
    fill_value: any = None,
):
    """
    根据模式创建并填充数组，或者仅创建空数组。

    Args:
        mode: 运行模式，可以是 "normal", "njit" 或 "cuda"。
        shape: 数组的形状。
        dtype: 数组的数据类型。
        fill_value: 用于填充数组的值。仅在 fill=True 时有效。
        fill: 一个布尔值，如果为 True，则创建并填充数组；如果为 False，则仅创建未初始化的空数组。
    """
    if mode in ["normal", "njit"]:
        if fill:
            return np.full(shape, fill_value, dtype=dtype)
        else:
            return np.empty(shape, dtype=dtype)
    elif mode == "cuda":
        d_array = nb.cuda.device_array(shape, dtype)
        if fill:
            d_array[:] = fill_value
        return d_array
    else:
        raise ValueError(f"Invalid mode: {mode}")


def get_max_temp_float_num(indicators_spec: dict, min_temp_float_num: int) -> int:
    """
    计算所有指标中最大的临时浮点数数量。
    """
    if not indicators_spec:
        # 如果字典为空，则返回最小值
        return min_temp_float_num

    max_from_specs = max(spec.get("temp_count", 0) for spec in indicators_spec.values())

    # 确保返回结果不小于设定的最小值
    return max(max_from_specs, min_temp_float_num)


def create_indicator_results(
    mode,
    indicators_spec: dict,
    indicator_enabled: np.ndarray,
    tohlcv_rows: int,
    min_rows: int,
    conf_count: int,
    dtype_dict: dict,
):
    """
    根据指标规格动态创建指标结果数组。
    """

    np_int_type = dtype_dict["np"]["int"]
    np_float_type = dtype_dict["np"]["float"]
    np_bool_type = dtype_dict["np"]["bool"]

    # 动态创建指标结果数组
    indicator_results = []
    for spec in indicators_spec.values():
        indicator_id = spec["id"]
        output_dim = spec["result_count"]

        # 根据指标是否启用确定行数
        rows = tohlcv_rows if indicator_enabled[indicator_id] else min_rows

        # 定义数组形状并创建数组
        shape = (conf_count, rows, output_dim)
        result_array = create_array(mode, shape, np_float_type)

        indicator_results.append(result_array)

    return tuple(indicator_results)


def initialize_outputs(
    mode,
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

    np_int_type = dtype_dict["np"]["int"]
    np_float_type = dtype_dict["np"]["float"]
    np_bool_type = dtype_dict["np"]["bool"]

    tohlcv_shape = tohlcv.shape
    tohlcv2_shape = tohlcv2.shape
    tohlcv_rows = tohlcv_shape[0]
    tohlcv2_rows = tohlcv2_shape[0]

    tohlcv_smooth_shape = tohlcv_shape
    tohlcv_smooth2_shape = tohlcv2_shape
    tohlcv_smooth = create_array(mode, tohlcv_smooth_shape, np_float_type)
    tohlcv_smooth2 = create_array(mode, tohlcv_smooth2_shape, np_float_type)

    signal_output_dim = signal_result_count
    backtest_output_dim = backtest_result_count

    temp_float_num = get_max_temp_float_num(indicators_spec, temp_float_num)

    # --- Indicator Result Arrays ---
    indicator_result = create_indicator_results(
        mode,
        indicators_spec,
        indicator_enabled,
        tohlcv_rows,
        min_rows,
        conf_count,
        dtype_dict,
    )

    indicator_result2 = create_indicator_results(
        mode,
        indicators_spec,
        indicator_enabled2,
        tohlcv2_rows,
        min_rows,
        conf_count,
        dtype_dict,
    )

    # --- Signal Result Arrays ---
    signal_shape = (conf_count, tohlcv_rows, signal_output_dim)
    signal_result = create_array(mode, signal_shape, np_bool_type)

    # --- Backtest Result Array ---
    backtest_shape = (conf_count, tohlcv_rows, backtest_output_dim)
    backtest_result = create_array(mode, backtest_shape, np_float_type)

    # --- Temporary Arrays ---
    int_temp_shape = (conf_count, tohlcv_rows, temp_int_num)
    int_temp_array = create_array(mode, int_temp_shape, np_int_type)

    float_temp_shape = (conf_count, tohlcv_rows, temp_float_num)
    float_temp_array = create_array(mode, float_temp_shape, np_float_type)

    bool_temp_shape = (conf_count, tohlcv_rows, temp_bool_num)
    bool_temp_array = create_array(mode, bool_temp_shape, np_bool_type)

    # --- Temporary Arrays ---
    int_temp_shape2 = (conf_count, tohlcv2_rows, temp_int_num)
    int_temp_array2 = create_array(mode, int_temp_shape2, np_int_type)

    float_temp_shape2 = (conf_count, tohlcv2_rows, temp_float_num)
    float_temp_array2 = create_array(mode, float_temp_shape2, np_float_type)

    bool_temp_shape2 = (conf_count, tohlcv2_rows, temp_bool_num)
    bool_temp_array2 = create_array(mode, bool_temp_shape2, np_bool_type)

    temp_args = (
        int_temp_array,
        int_temp_array2,
        float_temp_array,
        float_temp_array2,
        bool_temp_array,
        bool_temp_array2,
    )

    return (
        tohlcv_smooth,
        tohlcv_smooth2,
        indicator_result,
        indicator_result2,
        signal_result,
        backtest_result,
        temp_args,
    )


def unpack_params(outputs, inputs):
    (
        tohlcv_smooth,
        tohlcv_smooth2,
        indicator_result,
        indicator_result2,
        signal_result,
        backtest_result,
        temp_args,
    ) = outputs
    (
        tohlcv,
        tohlcv2,
        mapping_data,
        indicator_params,
        indicator_params2,
        indicator_enabled,
        indicator_enabled2,
        signal_params,
        backtest_params,
    ) = inputs
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

    cpu_params = (data_args, indicator_args, signal_args, backtest_args, temp_args)
    return cpu_params


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
    (
        int_temp_array,
        int_temp_array2,
        float_temp_array,
        float_temp_array2,
        bool_temp_array,
        bool_temp_array2,
    ) = temp_args

    (sma_params, sma2_params, bbands_params, atr_params, psar_params) = indicator_params
    (sma_params2, sma2_params2, bbands_params2, atr_params2, psar_params2) = (
        indicator_params2
    )
    (sma_result, sma2_result, bbands_result, atr_result, psar_result) = indicator_result
    (sma_result2, sma2_result2, bbands_result2, atr_result2, psar_result2) = (
        indicator_result2
    )

    indicator_params_child = (
        sma_params[idx],
        sma2_params[idx],
        bbands_params[idx],
        atr_params[idx],
        psar_params[idx],
    )
    indicator_params2_child = (
        sma_params2[idx],
        sma2_params2[idx],
        bbands_params2[idx],
        atr_params2[idx],
        psar_params2[idx],
    )
    indicator_result_child = (
        sma_result[idx],
        sma2_result[idx],
        bbands_result[idx],
        atr_result[idx],
        psar_result[idx],
    )
    indicator_result2_child = (
        sma_result2[idx],
        sma2_result2[idx],
        bbands_result2[idx],
        atr_result2[idx],
        psar_result2[idx],
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
        int_temp_array2[idx],
        float_temp_array[idx],
        float_temp_array2[idx],
        bool_temp_array[idx],
        bool_temp_array2[idx],
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
    (
        int_temp_array,
        int_temp_array2,
        float_temp_array,
        float_temp_array2,
        bool_temp_array,
        bool_temp_array2,
    ) = temp_args

    return {
        "tohlcv": tohlcv,
        "tohlcv2": tohlcv2,
        "mapping_data": mapping_data,
        "indicator_result": indicator_result,
        "indicator_result2": indicator_result2,
        "signal_result": signal_result,
        "backtest_result": backtest_result,
        "int_temp_array": int_temp_array,
        "int_temp_array2": int_temp_array2,
        "float_temp_array": float_temp_array,
        "float_temp_array2": float_temp_array2,
        "bool_temp_array": bool_temp_array,
        "bool_temp_array2": bool_temp_array2,
    }


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
