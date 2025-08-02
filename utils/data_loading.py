import numpy as np
import numba as nb
import pandas as pd
from utils.data_types import get_numba_data_types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

tohlcv_name = ["time", "open", "high", "low", "close", "volume"]


default_dtype_dict = get_numba_data_types(enable64=True)


def load_tohlcv_from_csv(
    file_path: str, data_size: int = None, dtype_dict=default_dtype_dict
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


def convert_tohlcv_numpy(df, dtype_dict=default_dtype_dict):
    return df[tohlcv_name].to_numpy().astype(dtype_dict["np"]["float"])


def transform_data_recursive(data, mode="to_device"):
    """
    递归地根据模式转换嵌套的元组、列表和数组。

    Args:
        data: 待转换的嵌套数据结构（元组、列表或 NumPy/CUDA 数组）。
        mode (str): 转换模式，可选值为 'to_device' (传输到 GPU) 或 'to_host' (拷贝回 CPU)。

    Returns:
        转换后的数据结构。
    """
    if mode not in ["to_device", "to_host"]:
        raise ValueError("mode 参数必须是 'to_device' 或 'to_host'")

    if isinstance(data, (tuple, list)):
        # 如果是元组或列表，递归处理其内部元素
        return type(data)(transform_data_recursive(item, mode=mode) for item in data)
    elif mode == "to_device" and isinstance(data, np.ndarray):
        # 如果是 NumPy 数组且模式为 'to_device'，传输到 CUDA 设备
        return nb.cuda.to_device(data)
    elif mode == "to_host" and isinstance(data, DeviceNDArray):
        # 如果是 CUDA 设备数组且模式为 'to_host'，拷贝回 CPU
        return data.copy_to_host()
    else:
        # 其他情况（如标量，或者不符合当前模式的数组类型），直接返回
        return data
