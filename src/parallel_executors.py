import numpy as np
import numba as nb
from src.core_logic import parallel_calc
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types, get_signature
from utils.time_utils import time_wrapper


def cpu_parallel_calc_jit(
    cache=True,
    dtype_dict=default_types,
):
    mode = "jit"
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    signature = get_signature(nb_int_type, nb_float_type)

    _parallel_calc = parallel_calc(mode, cache=cache, dtype_dict=dtype_dict)

    def _cpu_parallel_calc_jit(params):
        (micro_data, micro_input, macro_data, macro_input,
         backtest_input) = params
        (micro_tohlcv, micro_tohlcv_smooth) = micro_data
        (micro_indicator_params, micro_signal_params, micro_indicator_result,
         micro_signal_result) = micro_input
        (macro_tohlcv, macro_tohlcv_smooth) = macro_data
        (macro_indicator_params, macro_signal_params, macro_indicator_result,
         macro_signal_result) = macro_input
        (backtest_params, backtest_result, temp_arrays) = backtest_input

        for idx in nb.prange(micro_indicator_params.shape[0]):
            micro_input_child = (micro_indicator_params[idx],
                                 micro_signal_params[idx],
                                 micro_indicator_result[idx],
                                 micro_signal_result[idx])
            macro_input_child = (macro_indicator_params[idx],
                                 macro_signal_params[idx],
                                 macro_indicator_result[idx],
                                 macro_signal_result[idx])
            backtest_input_child = (backtest_params[idx], backtest_result[idx],
                                    temp_arrays[idx])
            params_child = (micro_data, micro_input_child, macro_data,
                            macro_input_child, backtest_input_child)
            _parallel_calc(params_child)

    return numba_wrapper(mode,
                         signature=signature,
                         cache_enabled=cache,
                         parallel=True)(_cpu_parallel_calc_jit)


@time_wrapper
def cpu_parallel_calc_jit_wrapper(*args, **kargs):
    return cpu_parallel_calc_jit(*args, **kargs)


def cpu_parallel_calc_njit(cache=True, dtype_dict=default_types):
    mode = "njit"
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    signature = get_signature(nb_int_type, nb_float_type)

    _parallel_calc = parallel_calc(mode, cache=cache, dtype_dict=dtype_dict)

    def _cpu_parallel_calc_njit(params):
        (micro_data, micro_input, macro_data, macro_input,
         backtest_input) = params
        (micro_tohlcv, micro_tohlcv_smooth) = micro_data
        (micro_indicator_params, micro_signal_params, micro_indicator_result,
         micro_signal_result) = micro_input
        (macro_tohlcv, macro_tohlcv_smooth) = macro_data
        (macro_indicator_params, macro_signal_params, macro_indicator_result,
         macro_signal_result) = macro_input
        (backtest_params, backtest_result, temp_arrays) = backtest_input

        for idx in nb.prange(micro_indicator_params.shape[0]):
            micro_input_child = (micro_indicator_params[idx],
                                 micro_signal_params[idx],
                                 micro_indicator_result[idx],
                                 micro_signal_result[idx])
            macro_input_child = (macro_indicator_params[idx],
                                 macro_signal_params[idx],
                                 macro_indicator_result[idx],
                                 macro_signal_result[idx])
            backtest_input_child = (backtest_params[idx], backtest_result[idx],
                                    temp_arrays[idx])
            params_child = (micro_data, micro_input_child, macro_data,
                            macro_input_child, backtest_input_child)
            _parallel_calc(params_child)

    return numba_wrapper(mode,
                         signature=signature,
                         cache_enabled=cache,
                         parallel=True)(_cpu_parallel_calc_njit)


@time_wrapper
def cpu_parallel_calc_njit_wrapper(*args, **kargs):
    return cpu_parallel_calc_njit(*args, **kargs)


def gpu_kernel_device(cache=True,
                      dtype_dict=default_types,
                      max_registers=None):
    mode = "cuda"
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    signature = get_signature(nb_int_type, nb_float_type)

    _parallel_calc = parallel_calc(mode, cache=cache, dtype_dict=dtype_dict)

    def _gpu_kernel_device(params):
        (micro_data, micro_input, macro_data, macro_input,
         backtest_input) = params
        (micro_tohlcv, micro_tohlcv_smooth) = micro_data
        (micro_indicator_params, micro_signal_params, micro_indicator_result,
         micro_signal_result) = micro_input
        (macro_tohlcv, macro_tohlcv_smooth) = macro_data
        (macro_indicator_params, macro_signal_params, macro_indicator_result,
         macro_signal_result) = macro_input
        (backtest_params, backtest_result, temp_arrays) = backtest_input

        _shape = micro_indicator_params.shape

        # 获取当前线程的唯一ID（起始索引）
        start_idx = nb.cuda.grid(1)
        # 获取所有启动线程的总数（步长）
        stride = nb.cuda.gridsize(1)

        # 步长循环：确保所有任务都被处理。
        # 当任务数多于线程数时，简单的处理方式会遗漏任务。
        # 这个循环让每个线程跳着处理属于自己的任务，保证所有任务都被覆盖且不重复。
        for idx in range(start_idx, _shape[0], stride):
            micro_input_child = (micro_indicator_params[idx],
                                 micro_signal_params[idx],
                                 micro_indicator_result[idx],
                                 micro_signal_result[idx])
            macro_input_child = (macro_indicator_params[idx],
                                 macro_signal_params[idx],
                                 macro_indicator_result[idx],
                                 macro_signal_result[idx])
            backtest_input_child = (backtest_params[idx], backtest_result[idx],
                                    temp_arrays[idx])
            params_child = (micro_data, micro_input_child, macro_data,
                            macro_input_child, backtest_input_child)
            _parallel_calc(params_child)

    return numba_wrapper(mode,
                         signature=signature,
                         cache_enabled=cache,
                         parallel=True,
                         max_registers=max_registers)(_gpu_kernel_device)


@time_wrapper
def gpu_kernel_device_wrapper(*args, **kargs):
    return gpu_kernel_device(*args, **kargs)
