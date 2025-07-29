import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest
from Test.conftest import df_data, np_data, dtype_dict
from Test.test_utils import assert_indicator_same
from utils.config_utils import get_params
from src.interface import calculate
from src.indicators.sma import sma_id, sma2_id, sma_name, sma2_name, sma_spec, sma2_spec


def test_accuracy(df_data,
                  np_data,
                  dtype_dict,
                  talib=False,
                  assert_func=assert_indicator_same):
    '''
    测试 SMA 指标的准确性, 以talib为准
    '''

    time = np_data[:, 0]
    open = np_data[:, 1]
    high = np_data[:, 2]
    low = np_data[:, 3]
    close = np_data[:, 4]
    volume = np_data[:, 5]

    close_series = df_data["close"]

    params_array = [[10], [20], [30]]

    for params in params_array:
        sma_period = params[0]

        params = get_params(num=1,
                            indicator_update={
                                sma_name: [params],
                            },
                            indicator_enabled={sma_id: True},
                            dtype_dict=dtype_dict)

        (indicator_result, indicator_result2, signal_result, backtest_result,
         int_temp_array, float_temp_array, bool_temp_array) = calculate(
             "njit",
             np_data,
             params["indicator_params"],
             params["indicator_enabled"],
             params["signal_params"],
             params["backtest_params"],
             cache=False,
             dtype_dict=dtype_dict,
         )

        sma_cpu_result = indicator_result[sma_id][0][:, 0]

        pandas_sma_results = ta.sma(close_series,
                                    length=int(sma_period),
                                    talib=talib)

        assert_func(sma_cpu_result, pandas_sma_results, sma_spec["ori_name"],
                    f"period {sma_period}")


def test_accuracy_talib(df_data,
                        np_data,
                        dtype_dict,
                        talib=True,
                        assert_func=assert_indicator_same):
    test_accuracy(df_data, np_data, dtype_dict, talib, assert_func)


def test_pandas_ta_and_talib_sma_same(df_data, dtype_dict):
    """
    比较 pandas-ta 和 talib 计算的 SMA 结果是否相同。
    预期结果是相同，所以使用 assert_indicator_same。
    """
    close_series = df_data["close"]

    params_array = [[10], [20], [30]]

    for p in params_array:
        sma_period = p[0]
        # 使用 pandas_ta 计算 SMA
        pandas_sma = ta.sma(close_series, length=int(sma_period), talib=False)

        # 使用 talib 计算 SMA
        talib_sma = ta.sma(close_series, length=int(sma_period), talib=True)

        # 使用 assert_indicator_same 验证两者是否相同
        assert_indicator_same(pandas_sma, talib_sma, sma_spec["ori_name"],
                              f"period {sma_period}")
