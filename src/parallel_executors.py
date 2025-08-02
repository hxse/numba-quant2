import numpy as np
import numba as nb
from src.core_logic import core_calc
from utils.data_types import get_params_signature
from utils.numba_unpack import unpack_params_child, get_conf_count


from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper


dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]

params_signature = get_params_signature(nb_int_type, nb_float_type, nb_bool_type)
signature = nb.void(params_signature)


if nb_params["mode"] == "normal":

    @nb_wrapper(
        mode=nb_params["mode"],
        signature=signature,
        cache_enabled=nb_params.get("cache", True),
        parallel=True,
    )
    def parallel_calc(params):
        (data_args, indicator_args, signal_args, backtest_args, temp_args) = params

        (
            indicator_params,
            indicator_params2,
            indicator_enabled,
            indicator_enabled2,
            indicator_result,
            indicator_result2,
        ) = indicator_args

        conf_count = get_conf_count(params)

        for idx in nb.prange(conf_count):
            _indicator_args = (
                indicator_params,
                indicator_params2,
                indicator_enabled,
                indicator_enabled2,
                indicator_result,
                indicator_result2,
            )
            _params = (
                data_args,
                _indicator_args,
                signal_args,
                backtest_args,
                temp_args,
            )
            _params_child = unpack_params_child(_params, idx)

            core_calc(_params_child)

elif nb_params["mode"] == "njit":

    @nb_wrapper(
        mode=nb_params["mode"],
        signature=signature,
        cache_enabled=nb_params.get("cache", True),
        parallel=True,
    )
    def parallel_calc(params):
        (data_args, indicator_args, signal_args, backtest_args, temp_args) = params

        (
            indicator_params,
            indicator_params2,
            indicator_enabled,
            indicator_enabled2,
            indicator_result,
            indicator_result2,
        ) = indicator_args

        conf_count = get_conf_count(params)

        for idx in nb.prange(conf_count):
            _indicator_args = (
                indicator_params,
                indicator_params2,
                indicator_enabled,
                indicator_enabled2,
                indicator_result,
                indicator_result2,
            )
            _params = (
                data_args,
                _indicator_args,
                signal_args,
                backtest_args,
                temp_args,
            )
            _params_child = unpack_params_child(_params, idx)

            core_calc(_params_child)


elif nb_params["mode"] == "cuda":

    @nb_wrapper(
        mode=nb_params["mode"],
        signature=signature,
        cache_enabled=nb_params.get("cache", True),
        parallel=True,
        max_registers=nb_params.get("max_registers", 24),
    )
    def parallel_calc(params):
        (data_args, indicator_args, signal_args, backtest_args, temp_args) = params

        (
            indicator_params,
            indicator_params2,
            indicator_enabled,
            indicator_enabled2,
            indicator_result,
            indicator_result2,
        ) = indicator_args

        conf_count = get_conf_count(params)

        # 获取当前线程的唯一ID（起始索引）
        start_idx = nb.cuda.grid(1)
        # 获取所有启动线程的总数（步长）
        stride = nb.cuda.gridsize(1)

        # 步长循环：确保所有任务都被处理。
        # 当任务数多于线程数时，简单的处理方式会遗漏任务。
        # 这个循环让每个线程跳着处理属于自己的任务，保证所有任务都被覆盖且不重复。
        for idx in range(start_idx, conf_count, stride):
            _indicator_args = (
                indicator_params,
                indicator_params2,
                indicator_enabled,
                indicator_enabled2,
                indicator_result,
                indicator_result2,
            )
            _params = (
                data_args,
                _indicator_args,
                signal_args,
                backtest_args,
                temp_args,
            )
            _params_child = unpack_params_child(_params, idx)

            core_calc(_params_child)
