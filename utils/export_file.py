import pandas as pd
import numpy as np
from pathlib import Path


def export_csv(
    name,
    np_data_obj,
    np_data2_obj,
    indicator_result_obj,
    indicator_result2_obj,
    signal_result_obj,
    backtest_result_obj,
    indicator_enabled,
    indicator_enabled2,
    index=0,
    write_csv=False,
):
    root_path = Path(f"output/{name}")
    root_path.mkdir(exist_ok=True)

    (columns, np_data) = np_data_obj
    tohlcv_df = pd.DataFrame(np_data, columns=columns)
    _name = f"{root_path}/tohlcv.csv"
    if write_csv:
        tohlcv_df.to_csv(_name, index=False)

    (columns, np_data2) = np_data2_obj
    tohlcv2_df = pd.DataFrame(np_data2, columns=columns)
    _name = f"{root_path}/tohlcv2.csv"
    if write_csv:
        tohlcv2_df.to_csv(_name, index=False)

    (columns, indicator_result) = indicator_result_obj
    _res = [indicator_result[k][index] for k, v in enumerate(indicator_enabled) if v]
    _res = np.hstack(_res)
    _col = [i for i in columns for i in i]
    indicator_df = pd.DataFrame(_res, columns=_col)
    _name = f"{root_path}/indicator.csv"
    if write_csv:
        indicator_df.to_csv(_name, index=False)

    (columns, indicator_result2) = indicator_result2_obj
    _res = [indicator_result2[k][index] for k, v in enumerate(indicator_enabled2) if v]
    _res = np.hstack(_res)
    _col = [i for i in columns for i in i]
    indicator2_df = pd.DataFrame(_res, columns=_col)
    _name = f"{root_path}/indicator2.csv"
    if write_csv:
        indicator2_df.to_csv(_name, index=False)

    (columns, signal_result) = signal_result_obj
    signal_df = pd.DataFrame(signal_result[index], columns=columns)
    _name = f"{root_path}/signal.csv"
    if write_csv:
        signal_df.to_csv(_name, index=False)

    (columns, backtest_result) = backtest_result_obj
    backtest_df = pd.DataFrame(backtest_result[index], columns=columns)
    _name = f"{root_path}/backtest.csv"
    if write_csv:
        backtest_df.to_csv(_name, index=False)

    return {
        "tohlcv_df": tohlcv_df,
        "tohlcv2_df": tohlcv2_df,
        "indicator_df": indicator_df,
        "indicator2_df": indicator2_df,
        "signal_df": signal_df,
        "backtest_df": backtest_df,
    }
