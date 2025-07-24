import numpy as np
import numba as nb
from src.parallel_executors import cpu_parallel_calc_jit, cpu_parallel_calc_njit, gpu_kernel_device, cpu_parallel_calc_jit_wrapper, cpu_parallel_calc_njit_wrapper, gpu_kernel_device_wrapper
from utils.numba_gpu_utils import auto_tune_cuda_parameters  # 导入新的工具函数
from utils.time_utils import time_wrapper
from utils.data_types import default_types, get_signature
from indicators.indicators_spec import flat_results_placeholders


def calculate(
        micro_tohlcv,
        micro_indicator_params,
        micro_signal_params,
        macro_tohlcv,
        macro_indicator_params,
        macro_signal_params,
        backtest_params,
        mode=None,
        cache=True,
        dtype_dict=default_types,
        temp_num=6,
        core_time=False,
        auto_tune_cuda_config=True,
        cuda_tuning_params={},  # 收集所有传递给 auto_tune_cuda_parameters 的参数
):
    """通用计算的核心逻辑，返回结果"""
    micro_tohlcv_smooth = np.full(micro_tohlcv.shape,
                                  np.nan,
                                  dtype=dtype_dict["np"]["float"])

    macro_tohlcv_smooth = np.full(macro_tohlcv.shape,
                                  np.nan,
                                  dtype=dtype_dict["np"]["float"])

    indicator_result_length = len(flat_results_placeholders)

    micro_indicator_result = np.full(
        (micro_indicator_params.shape[0], micro_tohlcv.shape[0],
         indicator_result_length),
        np.nan,
        dtype=dtype_dict["np"]["float"])

    macro_indicator_result = np.full(
        (micro_indicator_params.shape[0], micro_tohlcv.shape[0],
         indicator_result_length),
        np.nan,
        dtype=dtype_dict["np"]["float"])

    micro_signal_result = np.full(
        (micro_indicator_params.shape[0], macro_tohlcv.shape[0], 10),
        np.nan,
        dtype=dtype_dict["np"]["float"])

    macro_signal_result = np.full(
        (micro_indicator_params.shape[0], macro_tohlcv.shape[0], 10),
        np.nan,
        dtype=dtype_dict["np"]["float"])

    backtest_result = np.full(
        (micro_indicator_params.shape[0], macro_tohlcv.shape[0], 10),
        np.nan,
        dtype=dtype_dict["np"]["float"])

    temp_arrays = np.full(
        (micro_indicator_params.shape[0], micro_tohlcv.shape[0], temp_num),
        np.nan,
        dtype=dtype_dict["np"]["float"])

    micro_data = (micro_tohlcv, micro_tohlcv_smooth)
    micro_input = (micro_indicator_params, micro_signal_params,
                   micro_indicator_result, micro_signal_result)

    macro_data = (macro_tohlcv, macro_tohlcv_smooth)
    macro_input = (macro_indicator_params, macro_signal_params,
                   macro_indicator_result, macro_signal_result)

    backtest_input = (backtest_params, backtest_result, temp_arrays)

    cpu_params = (micro_data, micro_input, macro_data, macro_input,
                  backtest_input)

    def get_output(params):
        (micro_data, micro_input, macro_data, macro_input,
         backtest_input) = params
        return (micro_input[2], micro_input[3], macro_input[2], macro_input[3],
                backtest_input[1], backtest_input[2])

    if mode == "jit":
        _func = cpu_parallel_calc_jit_wrapper if core_time else cpu_parallel_calc_jit
        _cpu_parallel_calc = _func(cache=cache, dtype_dict=dtype_dict)
        _cpu_parallel_calc(cpu_params)
        return get_output(cpu_params)
    elif mode == "njit":
        _func = cpu_parallel_calc_njit_wrapper if core_time else cpu_parallel_calc_njit
        _cpu_parallel_calc = _func(cache=cache, dtype_dict=dtype_dict)
        _cpu_parallel_calc(cpu_params)
        return get_output(cpu_params)
    elif mode == "cuda":
        gpu_output = [[nb.cuda.to_device(i) for i in p] for p in cpu_params]

        if auto_tune_cuda_config:
            threadsperblock, blockspergrid, max_registers = auto_tune_cuda_parameters(
                workload_size=micro_indicator_params.shape[0],
                **cuda_tuning_params  # 将所有额外的参数传递给自动调优函数
            )
        else:
            threadsperblock = 256
            blockspergrid = (micro_indicator_params.shape[0] +
                             (threadsperblock - 1)) // threadsperblock
            if blockspergrid == 0:
                blockspergrid = 1
            max_registers = None  # 保持默认值或根据您的需求设置

        _func = gpu_kernel_device_wrapper if core_time else gpu_kernel_device
        _gpu_kernel_device = _func(cache=cache,
                                   dtype_dict=dtype_dict,
                                   max_registers=max_registers)
        _gpu_kernel_device[blockspergrid, threadsperblock](gpu_output)
        nb.cuda.synchronize()

        return tuple(i.copy_to_host() for i in get_output(gpu_output))
    else:
        raise ValueError(f"Invalid mode: {mode}")


@time_wrapper
def calculate_time_wrapper(*args, **kwargs):
    return calculate(*args, **kwargs)
