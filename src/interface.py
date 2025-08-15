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

from utils.numba_params import nb_params
from utils.outputs_global import get_outputs_from_global, set_outputs_from_global

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
    dtype_dict=default_dtype_dict,
    temp_int_num=1,
    temp_float_num=4,
    temp_bool_num=4,
    min_rows=0,  # 最小填充数组行数
    core_time=False,
    auto_tune_cuda_config=True,
    reuse_outputs=True,
    max_size=1,
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
        mapping_data = np.zeros(tohlcv.shape[0], dtype=dtype_dict["np"]["int"])

    lookup_dict = {
        "_conf_count": _conf_count,
        "tohlcv_shape": tohlcv.shape,
        "tohlcv2_shape": tohlcv2.shape,
        "temp_int_num": temp_int_num,
        "temp_float_num": temp_float_num,
        "temp_bool_num": temp_bool_num,
        "min_rows": min_rows,
    }
    outputs = None
    if reuse_outputs and max_size > 0:
        outputs = get_outputs_from_global(lookup_dict)

    if outputs:
        print("已复用上一次的结果数组,避免重复初始化开销")
    else:
        # 在gpu模式下,outputs会直接生成为gpu数组,数组太大了,省略转换,直接生成空数组
        outputs = initialize_outputs(
            mode,
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

    if reuse_outputs and max_size > 0 and outputs:
        set_outputs_from_global(lookup_dict, outputs, max_size=max_size)

    inputs = (
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

    if mode == "cuda":
        # inputs数组,在cuda模式下,会被转换成gpu,数组小,转换快
        inputs = transform_data_recursive(inputs, mode="to_device")
    params = unpack_params(outputs, inputs)

    end_time = time.perf_counter()
    print("数据生成时间:", end_time - start_time)

    if mode in ["normal", "njit"]:

        def _launch(_p):
            parallel_calc(_p)

        if core_time:
            timed_launch_func = time_wrapper(_launch)
            timed_launch_func(params)
        else:
            _launch(params)

        return get_output(params)
    elif mode == "cuda":
        if auto_tune_cuda_config:
            (threadsperblock, blockspergrid, max_registers) = auto_tune_cuda_parameters(
                register_per_thread=nb_params.get("max_registers", 24),
                workload_size=_conf_count,
            )
        else:
            threadsperblock = 256
            blockspergrid = (_conf_count + (threadsperblock - 1)) // threadsperblock
            if blockspergrid == 0:
                blockspergrid = 1
            max_registers = None  # 保持默认值或根据您的需求设置

        def _launch(_p):
            parallel_calc[blockspergrid, threadsperblock](_p)
            nb.cuda.synchronize()

        if core_time:
            timed_launch_func = time_wrapper(_launch)
            timed_launch_func(params)
        else:
            _launch(params)

        start_time = time.perf_counter()
        print("开始把数据从gpu提取到cpu")

        cpu_params = transform_data_recursive(get_output(params), mode="to_host")

        end_time = time.perf_counter()
        print("数据 gpu -> cpu:", end_time - start_time)
        return cpu_params
    else:
        raise ValueError(f"Invalid mode: {mode}")


@time_wrapper
def entry_func_wrapper(*args, **kwargs):
    return entry_func(*args, **kwargs)
