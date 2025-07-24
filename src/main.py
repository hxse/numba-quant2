import sys
from pathlib import Path

root_path = next((p for p in Path(__file__).resolve().parents
                  if (p / "pyproject.toml").is_file()), None)
if root_path:
    sys.path.insert(0, str(root_path))

from utils.data_loading import load_tohlcv_from_csv, convert_tohlcv_numpy
from utils.data_types import get_numba_data_types

import time

start_time = time.time()
import numpy as np
import pandas as pd
import pandas_ta as ta
import numba as nb
from src.interface import calculate, calculate_time_wrapper
from indicators.indicators_spec import flat_params_values

end_time = time.time()
print(f"numba模块导入冷启动时间: {end_time - start_time:.4f} 秒")


def prepare_inputs(micro_path,
                   macro_path,
                   data_size=None,
                   enable64=True,
                   num=200):
    dtype_dict = get_numba_data_types(enable64=enable64)

    micro_df = load_tohlcv_from_csv(micro_path, data_size, dtype_dict)
    micro_np = convert_tohlcv_numpy(micro_df, dtype_dict)

    macro_df = load_tohlcv_from_csv(macro_path, data_size, dtype_dict)
    macro_np = convert_tohlcv_numpy(macro_df, dtype_dict)

    micro_indicator_params = np.array([flat_params_values for i in range(num)],
                                      dtype=dtype_dict["np"]["float"])

    macro_indicator_params = np.array([flat_params_values for i in range(num)],
                                      dtype=dtype_dict["np"]["float"])

    micro_signal_params = np.array([[1.0 + i, 2.0 + i, 3.0 + i]
                                    for i in range(num)],
                                   dtype=dtype_dict["np"]["float"])

    macro_signal_params = np.array([[1.0 + i, 2.0 + i, 3.0 + i]
                                    for i in range(num)],
                                   dtype=dtype_dict["np"]["float"])

    backtest_params = np.array([[1.0 + i, 2.0 + i, 3.0 + i]
                                for i in range(num)],
                               dtype=dtype_dict["np"]["float"])
    return {
        "micro_df": micro_df,
        "micro_np": micro_np,
        "macro_df": macro_df,
        "macro_np": macro_np,
        "micro_indicator_params": micro_indicator_params,
        "macro_indicator_params": macro_indicator_params,
        "micro_signal_params": micro_signal_params,
        "macro_signal_params": macro_signal_params,
        "backtest_params": backtest_params,
        "dtype_dict": dtype_dict,
    }


def run(pre_run=True,
        cache=True,
        core_time=True,
        task_time=True,
        total_time=False):
    '''
    pre_run 控制是否执行第一次迭代 (预运行)
    total_time 包含jit,njit,cuda的完整运行时间,不包括csv文件导入时间
    task_time 包含数据预生成和内核运行的时间
    core_time 内核运行的时间
    '''
    print("#### CUDA 可用性检测 ####")
    if nb.cuda.is_available():
        print("CUDA 可用")
    else:
        print("CUDA 不可用")
        exit()

    for i in range(2):
        # 如果是第一次迭代 (i=0) 并且 pre_run 为 False，则跳过当前循环
        if i == 0 and not pre_run:
            continue

        micro_path = "database/live/BTC_USDT/15m/BTC_USDT_15m_20230228 160000.csv"
        macro_path = "database/live/BTC_USDT/4h/BTC_USDT_4h_20230228 160000.csv"

        data_size = 40 * 1000
        data_size = data_size if i == 0 else data_size + i

        result = prepare_inputs(micro_path,
                                macro_path,
                                data_size=data_size,
                                enable64=True,
                                num=200)

        print(f"\nstart task idx: {i}", "并发数量:",
              len(result["micro_indicator_params"]), "数据数量:",
              len(result["micro_np"]))

        start_time = time.time()

        mode_array = ["jit", "njit", "cuda"]
        # mode_array = ["jit"]

        for mode in mode_array:
            _func = calculate_time_wrapper if task_time else calculate
            (micro_indicator_result, micro_signal_result,
             macro_indicator_result, macro_signal_result, backtest_result,
             temp_arrays) = _func(result["micro_np"],
                                  result["micro_indicator_params"],
                                  result["micro_signal_params"],
                                  result["macro_np"],
                                  result["macro_indicator_params"],
                                  result["macro_signal_params"],
                                  result["backtest_params"],
                                  mode=mode,
                                  cache=cache,
                                  dtype_dict=result["dtype_dict"],
                                  core_time=core_time)

            if i != 0:
                print(f"{mode} out_arrays length:",
                      len(micro_indicator_result))
                print(f"{mode} out_arrays:", micro_indicator_result)
                print(f"{mode} temp_arrays length:", len(temp_arrays))
                print(f"{mode} temp_arrays:", temp_arrays)
                import pdb
                pdb.set_trace()

        if total_time:
            print(
                f"Task {i} total_time: {time.time() - start_time:.4f} seconds")


if __name__ == '__main__':
    run()
