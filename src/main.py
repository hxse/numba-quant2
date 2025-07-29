import sys
from pathlib import Path

root_path = next((p for p in Path(__file__).resolve().parents
                  if (p / "pyproject.toml").is_file()), None)
if root_path:
    sys.path.insert(0, str(root_path))

import time

start_time = time.time()
import numpy as np
import pandas as pd
import pandas_ta as ta
import numba as nb
from src.interface import calculate, calculate_time_wrapper

end_time = time.time()
print(f"numba模块导入冷启动时间: {end_time - start_time:.4f} 秒")

from utils.config_utils import get_dtype_dict, perpare_data, get_params
from src.indicators.sma import sma_id, sma2_id, sma_name, sma2_name, sma_spec, sma2_spec
from src.indicators.bbands import bbands_id, bbands_name, bbands_spec


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

        path = "database/live/BTC_USDT/15m/BTC_USDT_15m_20230228 160000.csv"
        path2 = "database/live/BTC_USDT/4h/BTC_USDT_4h_20230228 160000.csv"

        data_size = 40 * 1000
        data_size = 10 if i == 0 else data_size

        dtype_dict = get_dtype_dict(enable64=True)

        df_data, np_data = perpare_data(path,
                                        data_size=data_size,
                                        dtype_dict=dtype_dict)
        df_data2, np_data2 = perpare_data(path2,
                                          data_size=data_size,
                                          dtype_dict=dtype_dict)

        params = get_params(num=1,
                            indicator_update={
                                sma_name: [[14]],
                                sma2_name: [[50]],
                                bbands_name: [[20, 2.0]]
                            },
                            signal_params=[0, 0],
                            indicator_enabled={
                                sma_id: True,
                                sma2_id: True,
                                bbands_id: True
                            },
                            dtype_dict=dtype_dict)

        print(f"\nstart task idx: {i}", "并发数量:",
              len(params["backtest_params"]), "数据数量:", len(np_data))

        start_time = time.time()

        mode_array = ["normal", "njit", "cuda"]
        # mode_array = ["njit"]
        # mode_array = ["cuda"]

        for mode in mode_array:
            _func = calculate_time_wrapper if task_time else calculate
            (
                indicator_result, indicator_result2, signal_result,
                backtest_result, int_temp_array, float_temp_array,
                bool_temp_array
            ) = _func(
                mode,
                np_data,
                params["indicator_params"],
                params["indicator_enabled"],
                params["signal_params"],
                params["backtest_params"],
                #   tohlcv2=np_data2,
                #   indicator_params2=params["indicator_params2"],
                #   indicator_enabled2=params["indicator_enabled2"],
                # mapping_data=params["mapping_data"],
                cache=cache,
                dtype_dict=dtype_dict,
                core_time=core_time)

            if i != 0:
                print(f"{mode} out_arrays length:", len(backtest_result))
                print(f"{mode} indicator_result:", indicator_result)
                # print(f"{mode} indicator_result2:", indicator_result2)
                print(f"{mode} signal_result:", signal_result)

        if total_time:
            print(
                f"Task {i} total_time: {time.time() - start_time:.4f} seconds")


if __name__ == '__main__':
    run()
