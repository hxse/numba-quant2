import sys
from pathlib import Path

root_path = next((p for p in Path(__file__).resolve().parents
                  if (p / "pyproject.toml").is_file()), None)
if root_path:
    sys.path.insert(0, str(root_path))

from utils.data_loading import load_tohlcv_from_csv, convert_tohlcv_numpy
from utils.config_utils import get_dtype_dict
import pytest


@pytest.fixture(scope="module")
def dtype_dict():
    """
    提供数据类型字典，用于模块内的所有测试。
    """
    return get_dtype_dict(True)


@pytest.fixture(scope="module")
def df_data(dtype_dict):
    """
    加载并提供 Pandas DataFrame 格式的 OHLCV 数据，用于模块内的所有测试。
    """
    micro_path = "database/live/BTC_USDT/15m/BTC_USDT_15m_20230228 160000.csv"
    return load_tohlcv_from_csv(micro_path,
                                data_size=None,
                                dtype_dict=dtype_dict)


@pytest.fixture(scope="module")
def np_data(df_data, dtype_dict):
    """
    将 df_data 转换为 NumPy 格式并提供，用于模块内的所有测试。
    """
    return convert_tohlcv_numpy(df_data, dtype_dict=dtype_dict)
