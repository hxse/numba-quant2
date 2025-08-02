import numpy as np
import numba as nb
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from src.parallel_executors import (
    parallel_calc,
    # parallel_calc_normal,
    # parallel_calc_njit,
    # parallel_calc_cuda,
)
from utils.numba_gpu_utils import auto_tune_cuda_parameters  # 导入新的工具函数
from utils.time_utils import time_wrapper
from utils.data_types import get_numba_data_types
from utils.numba_unpack import unpack_params, get_output, initialize_outputs
from utils.data_loading import transform_data_recursive
import time

default_dtype_dict = get_numba_data_types(enable64=True)


def entry_func(
    mode,
    tohlcv,
    indicator_params,
    indicator_enabled,
    signal_params,
    backtest_params,
    tohlcv2=None,
    indicator_params2=None,
    indicator_enabled2=None,
    mapping_data=None,
    cache=False,
    dtype_dict=default_dtype_dict,
    min_rows=0,  # 最小填充数组行数
    temp_int_num=1,
    temp_float_num=1,
    temp_bool_num=4,
    core_time=False,
    auto_tune_cuda_config=True,
    cuda_tuning_params={},  # 收集所有传递给 auto_tune_cuda_parameters 的参数
):
    """
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
    """
    start_time = time.perf_counter()

    _conf_count = backtest_params.shape[0]

    if tohlcv2 is None:
        tohlcv2 = tohlcv[-min_rows:].copy()

    if indicator_params2 is None:
        indicator_params2 = tuple(i.copy() for i in indicator_params)

    if indicator_enabled2 is None:
        indicator_enabled2 = np.zeros_like(indicator_enabled)

    if mapping_data is None:
        # todo 待完善
        mapping_data = np.zeros_like(signal_params)

    outputs = initialize_outputs(
        tohlcv,
        tohlcv2,
        indicator_params,
        indicator_params2,
        indicator_enabled,
        indicator_enabled2,
        _conf_count,
        dtype_dict,
        temp_int_num=temp_int_num,
        temp_float_num=temp_float_num,
        temp_bool_num=temp_bool_num,
        min_rows=min_rows,
    )

    cpu_params = unpack_params(
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
    )

    if mode == "normal":
        end_time = time.perf_counter()
        print("数据生成时间:", end_time - start_time)

        def _launch(params):
            parallel_calc(params)

        if core_time:
            timed_launch_func = time_wrapper(_launch)
            timed_launch_func(cpu_params)
        else:
            _launch(cpu_params)

        return get_output(cpu_params)
    elif mode == "njit":
        end_time = time.perf_counter()
        print("数据生成时间:", end_time - start_time)

        def _launch(params):
            parallel_calc(params)

        if core_time:
            timed_launch_func = time_wrapper(_launch)
            timed_launch_func(cpu_params)
        else:
            _launch(cpu_params)
        return get_output(cpu_params)
    elif mode == "cuda":
        gpu_params = transform_data_recursive(cpu_params, mode="to_device")

        end_time = time.perf_counter()
        print("数据生成时间:", end_time - start_time)

        if auto_tune_cuda_config:
            threadsperblock, blockspergrid, max_registers = auto_tune_cuda_parameters(
                workload_size=_conf_count,
                **cuda_tuning_params,  # 将所有额外的参数传递给自动调优函数
            )
        else:
            threadsperblock = 256
            blockspergrid = (_conf_count + (threadsperblock - 1)) // threadsperblock
            if blockspergrid == 0:
                blockspergrid = 1
            max_registers = None  # 保持默认值或根据您的需求设置

        def _launch(params):
            parallel_calc[blockspergrid, threadsperblock](params)
            nb.cuda.synchronize()

        if core_time:
            timed_launch_func = time_wrapper(_launch)
            timed_launch_func(gpu_params)
        else:
            _launch(gpu_params)

        return transform_data_recursive(get_output(gpu_params), mode="to_host")
    else:
        raise ValueError(f"Invalid mode: {mode}")


@time_wrapper
def entry_func_wrapper(*args, **kwargs):
    return entry_func(*args, **kwargs)
