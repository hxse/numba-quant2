import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest
from Test.conftest import df_data, np_data, dtype_dict
from Test.test_utils import assert_indicator_same
from utils.config_utils import get_params
from src.interface import calculate
from src.indicators.bbands import bbands_id, bbands_name, bbands_spec


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

    params_array = [[10, 2.0], [20, 2.5], [30, 3.0]]

    for params in params_array:
        bbands_period = params[0]
        bbands_std_mult = params[1]

        params = get_params(num=1,
                            indicator_update={
                                bbands_name: [params],
                            },
                            indicator_enabled={bbands_id: True},
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

        middle_result = indicator_result[bbands_id][0][:, 0]
        upper_result = indicator_result[bbands_id][0][:, 1]
        lower_result = indicator_result[bbands_id][0][:, 2]

        # Pandas TA 计算布林带
        pandas_bbands = ta.bbands(close=close_series,
                                  length=bbands_period,
                                  std=bbands_std_mult,
                                  talib=talib)
        pandas_middle = pandas_bbands[
            f"BBM_{bbands_period}_{bbands_std_mult}"].values
        pandas_upper = pandas_bbands[
            f"BBU_{bbands_period}_{bbands_std_mult}"].values
        pandas_lower = pandas_bbands[
            f"BBL_{bbands_period}_{bbands_std_mult}"].values

        assert_func(middle_result, pandas_middle, bbands_spec["ori_name"],
                    f"period {bbands_period} std_mult {bbands_std_mult}")

        assert_func(upper_result, pandas_upper, bbands_spec["ori_name"],
                    f"period {bbands_period} std_mult {bbands_std_mult}")

        assert_func(lower_result, pandas_lower, bbands_spec["ori_name"],
                    f"period {bbands_period} std_mult {bbands_std_mult}")


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

    params_array = [[10, 2.0], [20, 2.5], [30, 3.0]]

    for params in params_array:
        bbands_period = params[0]
        bbands_std_mult = params[1]

        # Pandas TA 计算布林带
        pandas_bbands = ta.bbands(close=close_series,
                                  length=bbands_period,
                                  std=bbands_std_mult,
                                  talib=False)
        pandas_middle = pandas_bbands[
            f"BBM_{bbands_period}_{bbands_std_mult}"].values
        pandas_upper = pandas_bbands[
            f"BBU_{bbands_period}_{bbands_std_mult}"].values
        pandas_lower = pandas_bbands[
            f"BBL_{bbands_period}_{bbands_std_mult}"].values

        # Pandas TA 计算布林带
        pandas_bbands_talib = ta.bbands(close=close_series,
                                        length=bbands_period,
                                        std=bbands_std_mult,
                                        talib=True)
        pandas_middle_talib = pandas_bbands[
            f"BBM_{bbands_period}_{bbands_std_mult}"].values
        pandas_upper_talib = pandas_bbands[
            f"BBU_{bbands_period}_{bbands_std_mult}"].values
        pandas_lower_talib = pandas_bbands[
            f"BBL_{bbands_period}_{bbands_std_mult}"].values

        assert_indicator_same(
            pandas_middle_talib, pandas_middle, bbands_spec["ori_name"],
            f"period {bbands_period} std_mult {bbands_std_mult}")

        assert_indicator_same(
            pandas_upper_talib, pandas_upper, bbands_spec["ori_name"],
            f"period {bbands_period} std_mult {bbands_std_mult}")

        assert_indicator_same(
            pandas_lower_talib, pandas_lower, bbands_spec["ori_name"],
            f"period {bbands_period} std_mult {bbands_std_mult}")
