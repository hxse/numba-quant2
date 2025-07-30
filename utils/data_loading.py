import numpy as np
import pandas as pd
from utils.data_types import default_types

tohlcv_name = ["time", "open", "high", "low", "close", "volume"]


def load_tohlcv_from_csv(
    file_path: str, data_size: int = None, dtype_dict=default_types
) -> np.ndarray:
    """
    从 CSV 文件加载 OHLCV 数据并进行处理。

    Args:
        file_path (str): CSV 文件的路径。
        data_size (int): 需要加载的数据大小。

    Returns:
        np.ndarray: 处理后的 OHLCV 数据。
    """
    df = pd.read_csv(file_path)
    # 将 timestamp 列重命名为 time
    if "timestamp" in df.columns:
        df.rename(columns={"timestamp": "time"}, inplace=True)
    # 将数值列转换为 np_float
    numeric_cols = tohlcv_name
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(dtype_dict["np"]["float"])
    df["date"] = pd.to_datetime(df["time"], unit="ms")

    if data_size and len(df) > data_size:
        # df = df.iloc[:data_size]
        df = df.iloc[-data_size:]

    return df


def convert_tohlcv_numpy(df, dtype_dict=default_types):
    return df[tohlcv_name].to_numpy().astype(dtype_dict["np"]["float"])
