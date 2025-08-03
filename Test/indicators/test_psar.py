import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest
from Test.conftest import df_data, np_data, dtype_dict
from Test.test_utils import assert_indicator_same, assert_indicator_different
from utils.config_utils import get_params
from src.interface import entry_func
from src.indicators.psar import psar_id, psar_name, psar_spec


def test_accuracy(
    df_data, np_data, dtype_dict, talib=False, assert_func=assert_indicator_same
):
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

    params_array = [[0.02, 0.02, 0.2], [0.03, 0.03, 0.3], [0.04, 0.04, 0.4]]

    for params in params_array:
        af0 = params[0]
        af = params[1]
        max_af = params[2]

        params = get_params(
            num=1,
            indicator_update={
                psar_name: [params],
            },
            indicator_enabled={psar_id: True},
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

        psar_long_result = indicator_result[psar_id][0][:, 0]
        psar_short_result = indicator_result[psar_id][0][:, 1]
        psar_af_result = indicator_result[psar_id][0][:, 2]
        psar_reversal_result = indicator_result[psar_id][0][:, 3]

        # Pandas TA 计算 PSAR
        pandas_psar = ta.psar(
            high=high_series,
            low=low_series,
            close=close_series,
            af0=af0,
            af=af,
            max_af=max_af,
            talib=talib,
        )
        # pandas_ta 的 PSAR 返回 PSARl (Long) 和 PSARs (Short)
        pandas_psar_long = pandas_psar[f"PSARl_{af0}_{max_af}"].values
        pandas_psar_short = pandas_psar[f"PSARs_{af0}_{max_af}"].values
        pandas_psar_af = pandas_psar[f"PSARaf_{af0}_{max_af}"].values
        pandas_psar_reversal = pandas_psar[f"PSARr_{af0}_{max_af}"].values

        assert_func(
            psar_long_result,
            pandas_psar_long,
            psar_spec["ori_name"],
            f"af0 {af0} af {af} max_af {max_af}",
        )

        assert_func(
            psar_short_result,
            pandas_psar_short,
            psar_spec["ori_name"],
            f"af0 {af0} af {af} max_af {max_af}",
        )

        assert_func(
            psar_af_result,
            pandas_psar_af,
            psar_spec["ori_name"],
            f"af0 {af0} af {af} max_af {max_af}",
        )
        assert_func(
            psar_reversal_result,
            pandas_psar_reversal,
            psar_spec["ori_name"],
            f"af0 {af0} af {af} max_af {max_af}",
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

    params_array = [[0.02, 0.02, 0.2], [0.03, 0.03, 0.3], [0.04, 0.04, 0.4]]

    for params in params_array:
        af0 = params[0]
        af = params[1]
        max_af = params[2]

        pandas_psar = ta.psar(
            high=high_series,
            low=low_series,
            close=close_series,
            af0=af0,
            af=af,
            max_af=max_af,
            talib=False,
        )

        pandas_psar_long = pandas_psar[f"PSARl_{af0}_{max_af}"].values
        pandas_psar_short = pandas_psar[f"PSARs_{af0}_{max_af}"].values
        pandas_psar_af = pandas_psar[f"PSARaf_{af0}_{max_af}"].values
        pandas_psar_reversal = pandas_psar[f"PSARr_{af0}_{max_af}"].values

        pandas_psar_talib = ta.psar(
            high=high_series,
            low=low_series,
            close=close_series,
            af0=af0,
            af=af,
            max_af=max_af,
            talib=True,
        )

        pandas_psar_long_talib = pandas_psar_talib[f"PSARl_{af0}_{max_af}"].values
        pandas_psar_short_talib = pandas_psar_talib[f"PSARs_{af0}_{max_af}"].values
        pandas_psar_af_talib = pandas_psar_talib[f"PSARaf_{af0}_{max_af}"].values
        pandas_psar_reversal_talib = pandas_psar_talib[f"PSARr_{af0}_{max_af}"].values

        assert_func(
            pandas_psar_long,
            pandas_psar_long_talib,
            psar_spec["ori_name"],
            f"af0 {af0} af {af} max_af {max_af}",
        )

        assert_func(
            pandas_psar_short,
            pandas_psar_short_talib,
            psar_spec["ori_name"],
            f"af0 {af0} af {af} max_af {max_af}",
        )

        assert_func(
            pandas_psar_af,
            pandas_psar_af_talib,
            psar_spec["ori_name"],
            f"af0 {af0} af {af} max_af {max_af}",
        )

        assert_func(
            pandas_psar_reversal,
            pandas_psar_reversal_talib,
            psar_spec["ori_name"],
            f"af0 {af0} af {af} max_af {max_af}",
        )
