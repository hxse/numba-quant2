import numpy as np
import numba as nb
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from src.parallel_executors import cpu_parallel_calc_jit, cpu_parallel_calc_njit, gpu_kernel_device, cpu_parallel_calc_jit_wrapper, cpu_parallel_calc_njit_wrapper, gpu_kernel_device_wrapper
from utils.numba_gpu_utils import auto_tune_cuda_parameters  # 导入新的工具函数
from utils.time_utils import time_wrapper
from utils.data_types import default_types
from utils.numba_unpack import unpack_params, get_output, initialize_outputs
import utils.filter_log


def transform_data_recursive(data, mode='to_device'):
    """
    递归地根据模式转换嵌套的元组、列表和数组。

    Args:
        data: 待转换的嵌套数据结构（元组、列表或 NumPy/CUDA 数组）。
        mode (str): 转换模式，可选值为 'to_device' (传输到 GPU) 或 'to_host' (拷贝回 CPU)。

    Returns:
        转换后的数据结构。
    """
    if mode not in ['to_device', 'to_host']:
        raise ValueError("mode 参数必须是 'to_device' 或 'to_host'")

    if isinstance(data, (tuple, list)):
        # 如果是元组或列表，递归处理其内部元素
        return type(data)(transform_data_recursive(item, mode=mode)
                          for item in data)
    elif mode == 'to_device' and isinstance(data, np.ndarray):
        # 如果是 NumPy 数组且模式为 'to_device'，传输到 CUDA 设备
        return nb.cuda.to_device(data)
    elif mode == 'to_host' and isinstance(data, DeviceNDArray):
        # 如果是 CUDA 设备数组且模式为 'to_host'，拷贝回 CPU
        return data.copy_to_host()
    else:
        # 其他情况（如标量，或者不符合当前模式的数组类型），直接返回
        return data


def calculate(
        mode,
        tohlcv,
        indicator_params,
        indicator_enabled,
        signal_params,
        backtest_params,
        tohlcv2=None,
        indicator_params2=None,
        indicator_enabled2=None,
        cache=False,
        dtype_dict=default_types,
        min_rows=0,  #最小填充数组行数
        temp_num=6,
        core_time=False,
        auto_tune_cuda_config=True,
        cuda_tuning_params={},  # 收集所有传递给 auto_tune_cuda_parameters 的参数
):
    '''
    目前的设计来说,同一波并发,可以变的参数如下
    indicator_params,indicator_params2,backtest_params,都是二维数组
    同一波并发,不可变的参数如下
    signal_params,indicator_enabled,indicator_enabled2,都是一维数组
    (注意, indicator_enabled 和 indicator_enabled2 也会依赖 signal_params)
    为什么这样设计?
    1. indicator_enabled如果改变的话,指标结果数组就不等长了,这样不允许
    2. 如果indicator_enabled永远为全都启用,指标结果数组占内存太大了
    3. 如果想启用指标探索的话,探索用循环就行了,优化用并发,既然探索用了循环,indicator_enabled就没必要在并发中可变了,循环中可变已经够用了
    4. 我更倾向于盘感驱动式量化交易(符合人的交易直觉),而不是像机械学习那样大规模探索指标和策略(看起来太黑箱了),所以indicator_enabled就没必要在并发中可变了,循环中可变已经够用了
    5. 为什么只有一个signal_params,如果用两个signal_params,是为了指标信号探索过程中的不同周期组合,这个就太复杂了(像机械学习),我只需要简单的信号模版选择功能就行了(盘感驱动的量化交易)
    '''
    _conf_count = backtest_params.shape[0]

    if tohlcv2 is None:
        tohlcv2 = tohlcv[-min_rows:].copy()

    if indicator_params2 is None:
        indicator_params2 = tuple(i.copy() for i in indicator_params)

    if indicator_enabled2 is None:
        indicator_enabled2 = np.zeros_like(indicator_enabled)

    outputs = initialize_outputs(tohlcv,
                                 tohlcv2,
                                 indicator_params,
                                 indicator_params2,
                                 indicator_enabled,
                                 indicator_enabled2,
                                 _conf_count,
                                 dtype_dict,
                                 temp_num,
                                 min_rows=min_rows)

    cpu_params = unpack_params(outputs, tohlcv, tohlcv2, indicator_params,
                               indicator_params2, indicator_enabled,
                               indicator_enabled2, signal_params,
                               backtest_params)

    if mode == "jit" or mode == "normal":
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
        gpu_params = transform_data_recursive(cpu_params, mode="to_device")

        if auto_tune_cuda_config:
            threadsperblock, blockspergrid, max_registers = auto_tune_cuda_parameters(
                workload_size=_conf_count,
                **cuda_tuning_params  # 将所有额外的参数传递给自动调优函数
            )
        else:
            threadsperblock = 256
            blockspergrid = (_conf_count +
                             (threadsperblock - 1)) // threadsperblock
            if blockspergrid == 0:
                blockspergrid = 1
            max_registers = None  # 保持默认值或根据您的需求设置

        _func = gpu_kernel_device_wrapper if core_time else gpu_kernel_device
        _gpu_kernel_device = _func(cache=cache,
                                   dtype_dict=dtype_dict,
                                   max_registers=max_registers)
        _gpu_kernel_device[blockspergrid, threadsperblock](gpu_params)
        nb.cuda.synchronize()

        return transform_data_recursive(get_output(gpu_params), mode="to_host")
    else:
        raise ValueError(f"Invalid mode: {mode}")


@time_wrapper
def calculate_time_wrapper(*args, **kwargs):
    return calculate(*args, **kwargs)
