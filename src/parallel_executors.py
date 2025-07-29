import numpy as np
import numba as nb
from src.core_logic import parallel_calc
from utils.numba_utils import numba_wrapper
from utils.data_types import default_types, get_params_signature
from utils.time_utils import time_wrapper
from utils.numba_unpack import unpack_params_child, get_conf_count


def cpu_parallel_calc_jit(
    cache=True,
    dtype_dict=default_types,
):
    '''
    工具函数版本运行失败
    '''
    mode = "jit"
    nb_int_type = dtype_dict["nb"]["int"]
    nb_float_type = dtype_dict["nb"]["float"]
    nb_bool_type = dtype_dict["nb"]["bool"]

    params_signature = get_params_signature(nb_int_type, nb_float_type,
                                            nb_bool_type)
    signature = nb.void(params_signature)

    _get_conf_count = get_conf_count(mode, cache=cache, dtype_dict=dtype_dict)
    _unpack_params_child = unpack_params_child(mode,
                                               cache=cache,
                                               dtype_dict=dtype_dict)

    _parallel_calc = parallel_calc(mode, cache=cache, dtype_dict=dtype_dict)

    def _cpu_parallel_calc_jit(params):

        (data_args, indicator_args, signal_args, backtest_args,
         temp_args) = params

        (indicator_params, indicator_params2, indicator_enabled,
         indicator_enabled2, indicator_result,
         indicator_result2) = indicator_args

        conf_count = _get_conf_count(params)

        for idx in nb.prange(conf_count):

            _indicator_args = (indicator_params, indicator_params2,
                               indicator_enabled, indicator_enabled2,
                               indicator_result, indicator_result2)
            _params = (data_args, _indicator_args, signal_args, backtest_args,
                       temp_args)
            _params_child = _unpack_params_child(_params, idx)

            _parallel_calc(_params_child)

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
    nb_bool_type = dtype_dict["nb"]["bool"]

    params_signature = get_params_signature(nb_int_type, nb_float_type,
                                            nb_bool_type)
    signature = nb.void(params_signature)

    _get_conf_count = get_conf_count(mode, cache=cache, dtype_dict=dtype_dict)
    _unpack_params_child = unpack_params_child(mode,
                                               cache=cache,
                                               dtype_dict=dtype_dict)

    _parallel_calc = parallel_calc(mode, cache=cache, dtype_dict=dtype_dict)

    def _cpu_parallel_calc_njit(params):

        (data_args, indicator_args, signal_args, backtest_args,
         temp_args) = params

        (indicator_params, indicator_params2, indicator_enabled,
         indicator_enabled2, indicator_result,
         indicator_result2) = indicator_args

        conf_count = _get_conf_count(params)

        for idx in nb.prange(conf_count):

            _indicator_args = (indicator_params, indicator_params2,
                               indicator_enabled, indicator_enabled2,
                               indicator_result, indicator_result2)
            _params = (data_args, _indicator_args, signal_args, backtest_args,
                       temp_args)
            _params_child = _unpack_params_child(_params, idx)

            _parallel_calc(_params_child)

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
    nb_bool_type = dtype_dict["nb"]["bool"]

    params_signature = get_params_signature(nb_int_type, nb_float_type,
                                            nb_bool_type)
    signature = nb.void(params_signature)

    _get_conf_count = get_conf_count(mode, cache=cache, dtype_dict=dtype_dict)
    _unpack_params_child = unpack_params_child(mode,
                                               cache=cache,
                                               dtype_dict=dtype_dict)

    _parallel_calc = parallel_calc(mode, cache=cache, dtype_dict=dtype_dict)

    def _gpu_kernel_device(params):

        (data_args, indicator_args, signal_args, backtest_args,
         temp_args) = params

        (indicator_params, indicator_params2, indicator_enabled,
         indicator_enabled2, indicator_result,
         indicator_result2) = indicator_args

        conf_count = _get_conf_count(params)

        # 获取当前线程的唯一ID（起始索引）
        start_idx = nb.cuda.grid(1)
        # 获取所有启动线程的总数（步长）
        stride = nb.cuda.gridsize(1)

        # 步长循环：确保所有任务都被处理。
        # 当任务数多于线程数时，简单的处理方式会遗漏任务。
        # 这个循环让每个线程跳着处理属于自己的任务，保证所有任务都被覆盖且不重复。
        for idx in range(start_idx, conf_count, stride):

            _indicator_args = (indicator_params, indicator_params2,
                               indicator_enabled, indicator_enabled2,
                               indicator_result, indicator_result2)
            _params = (data_args, _indicator_args, signal_args, backtest_args,
                       temp_args)
            _params_child = _unpack_params_child(_params, idx)

            _parallel_calc(_params_child)

    return numba_wrapper(mode,
                         signature=signature,
                         cache_enabled=cache,
                         parallel=True,
                         max_registers=max_registers)(_gpu_kernel_device)


@time_wrapper
def gpu_kernel_device_wrapper(*args, **kargs):
    return gpu_kernel_device(*args, **kargs)
