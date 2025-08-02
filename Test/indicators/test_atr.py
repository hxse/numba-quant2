import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest
from Test.conftest import df_data, np_data, dtype_dict
from Test.test_utils import assert_indicator_same, assert_indicator_different
from utils.config_utils import get_params
from src.interface import entry_func
from src.indicators.atr import atr_id, atr_name, atr_spec


def test_accuracy(
    df_data, np_data, dtype_dict, talib=False, assert_func=assert_indicator_different
):
    """
    测试 SMA 指标的准确性, 以talib为准
    """

    time = np_data[:, 0]
    open = np_data[:, 1]
    high = np_data[:, 2]
    low = np_data[:, 3]
    close = np_data[:, 4]
    volume = np_data[:, 5]

    time_series = df_data["time"]
    open_series = df_data["open"]
    high_series = df_data["high"]
    low_series = df_data["low"]
    close_series = df_data["close"]
    volume_series = df_data["volume"]

    params_array = [[10], [20], [30]]

    for params in params_array:
        atr_period = params[0]

        params = get_params(
            num=1,
            indicator_update={
                atr_name: [params],
            },
            indicator_enabled={atr_id: True},
            dtype_dict=dtype_dict,
        )

        (
            indicator_result,
            indicator_result2,
            signal_result,
            backtest_result,
            int_temp_array,
            float_temp_array,
            bool_temp_array,
        ) = entry_func(
            "njit",
            np_data,
            params["indicator_params"],
            params["indicator_enabled"],
            params["signal_params"],
            params["backtest_params"],
            cache=False,
            dtype_dict=dtype_dict,
        )

        atr_cpu_result = indicator_result[atr_id][0][:, 0]

        pandas_atr_results = ta.atr(
            high_series, low_series, close_series, length=int(atr_period), talib=talib
        )

        assert_func(
            atr_cpu_result,
            pandas_atr_results,
            atr_spec["ori_name"],
            f"period {atr_period}",
        )


def test_accuracy_talib(
    df_data, np_data, dtype_dict, talib=True, assert_func=assert_indicator_same
):
    test_accuracy(df_data, np_data, dtype_dict, talib, assert_func)


def test_pandas_ta_and_talib_atr_same(df_data, dtype_dict):
    """
    比较 pandas-ta 和 talib 计算的 SMA 结果是否相同。
    预期结果是相同，所以使用 assert_indicator_same。
    """
    time_series = df_data["time"]
    open_series = df_data["open"]
    high_series = df_data["high"]
    low_series = df_data["low"]
    close_series = df_data["close"]
    volume_series = df_data["volume"]

    params_array = [[10], [20], [30]]

    for p in params_array:
        atr_period = p[0]
        # 使用 pandas_ta 计算 SMA
        pandas_sma = ta.atr(
            high_series, low_series, close_series, length=int(atr_period), talib=False
        )

        # 使用 talib 计算 SMA
        talib_sma = ta.atr(
            high_series, low_series, close_series, length=int(atr_period), talib=True
        )

        # 使用 assert_indicator_different 验证两者是否不同
        assert_indicator_different(
            pandas_sma, talib_sma, atr_spec["ori_name"], f"period {atr_period}"
        )
