import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest
from Test.conftest import df_data, np_data, dtype_dict
from Test.test_utils import assert_indicator_same
from utils.config_utils import get_params
from src.interface import entry_func


from src.indicators.indicators_wrapper import indicators_spec

bbands_spec = indicators_spec["bbands"]
bbands_name = bbands_spec["name"]
bbands_id = bbands_spec["id"]


def test_accuracy(
    df_data, np_data, dtype_dict, talib=False, assert_func=assert_indicator_same
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

    params_array = [[10, 2.0], [20, 2.5], [30, 3.0]]

    for params in params_array:
        bbands_period = params[0]
        bbands_std_mult = params[1]

        params = get_params(
            num=1,
            indicator_update={
                bbands_name: [params],
            },
            indicator_enabled={bbands_name: True},
            dtype_dict=dtype_dict,
        )

        result = entry_func(
            "njit",
            np_data,
            params["indicator_params"],
            params["indicator_enabled"],
            params["signal_params"],
            params["backtest_params"],
            dtype_dict=dtype_dict,
            reuse_outputs=False,
        )

        middle_result = result["indicator_result"][bbands_id][0][:, 0]
        upper_result = result["indicator_result"][bbands_id][0][:, 1]
        lower_result = result["indicator_result"][bbands_id][0][:, 2]

        # Pandas TA 计算布林带
        pandas_bbands = ta.bbands(
            close=close_series, length=bbands_period, std=bbands_std_mult, talib=talib
        )
        pandas_middle = pandas_bbands[f"BBM_{bbands_period}_{bbands_std_mult}"].values
        pandas_upper = pandas_bbands[f"BBU_{bbands_period}_{bbands_std_mult}"].values
        pandas_lower = pandas_bbands[f"BBL_{bbands_period}_{bbands_std_mult}"].values

        assert_func(
            middle_result,
            pandas_middle,
            bbands_name,
            f"period {bbands_period} std_mult {bbands_std_mult}",
        )

        assert_func(
            upper_result,
            pandas_upper,
            bbands_name,
            f"period {bbands_period} std_mult {bbands_std_mult}",
        )

        assert_func(
            lower_result,
            pandas_lower,
            bbands_name,
            f"period {bbands_period} std_mult {bbands_std_mult}",
        )


def test_accuracy_talib(
    df_data, np_data, dtype_dict, talib=True, assert_func=assert_indicator_same
):
    test_accuracy(df_data, np_data, dtype_dict, talib, assert_func)


def test_pandas_ta_and_talib_sma_same(
    df_data, dtype_dict, assert_func=assert_indicator_same
):
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

    params_array = [[10, 2.0], [20, 2.5], [30, 3.0]]

    for params in params_array:
        bbands_period = params[0]
        bbands_std_mult = params[1]

        # Pandas TA 计算布林带
        pandas_bbands = ta.bbands(
            close=close_series, length=bbands_period, std=bbands_std_mult, talib=False
        )
        pandas_middle = pandas_bbands[f"BBM_{bbands_period}_{bbands_std_mult}"].values
        pandas_upper = pandas_bbands[f"BBU_{bbands_period}_{bbands_std_mult}"].values
        pandas_lower = pandas_bbands[f"BBL_{bbands_period}_{bbands_std_mult}"].values

        # Pandas TA 计算布林带
        pandas_bbands_talib = ta.bbands(
            close=close_series, length=bbands_period, std=bbands_std_mult, talib=True
        )
        pandas_middle_talib = pandas_bbands[
            f"BBM_{bbands_period}_{bbands_std_mult}"
        ].values
        pandas_upper_talib = pandas_bbands[
            f"BBU_{bbands_period}_{bbands_std_mult}"
        ].values
        pandas_lower_talib = pandas_bbands[
            f"BBL_{bbands_period}_{bbands_std_mult}"
        ].values

        assert_func(
            pandas_middle_talib,
            pandas_middle,
            bbands_name,
            f"period {bbands_period} std_mult {bbands_std_mult}",
        )

        assert_func(
            pandas_upper_talib,
            pandas_upper,
            bbands_name,
            f"period {bbands_period} std_mult {bbands_std_mult}",
        )

        assert_func(
            pandas_lower_talib,
            pandas_lower,
            bbands_name,
            f"period {bbands_period} std_mult {bbands_std_mult}",
        )
