from utils.data_loading import load_tohlcv_from_csv, convert_tohlcv_numpy
from utils.data_types import get_numba_data_types
import numpy as np

from src.indicators.sma import sma_id, sma2_id, sma_name, sma2_name, sma_spec, sma2_spec
from src.indicators.bbands import bbands_id, bbands_name, bbands_spec
from src.indicators.atr import atr_id, atr_name, atr_spec

indicator_count = atr_id + 1  # 最大的指标id值+1


def get_dtype_dict(enable64=True):
    return get_numba_data_types(enable64=enable64)


default_dtype_dict = get_dtype_dict(True)


def ensure_c_contiguous(v):
    if not v.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(v)
    return v


def get_params(
    num,
    indicator_update={},
    indicator_enabled={},
    indicator_update2={},
    indicator_enabled2={},
    signal_params=[],
    backtest_params=[],
    dtype_dict=default_dtype_dict,
):
    indicator_params = get_indicator_params(
        num, update_params=indicator_update, dtype_dict=dtype_dict
    )
    indicator_enabled = get_indicator_enabled(
        update_params=indicator_enabled, dtype_dict=dtype_dict
    )
    signal_params = get_signal_params(params=signal_params, dtype_dict=dtype_dict)
    backtest_params = get_backtest_params(
        num, params=backtest_params, dtype_dict=dtype_dict
    )

    indicator_params2 = get_indicator_params(
        num, update_params=indicator_update2, dtype_dict=dtype_dict
    )
    indicator_enabled2 = get_indicator_enabled(
        update_params=indicator_enabled2, dtype_dict=dtype_dict
    )

    # todo 待完善
    mapping_data = get_mapping_data([], [], dtype_dict=dtype_dict)

    return {
        "indicator_params": indicator_params,
        "indicator_enabled": indicator_enabled,
        "signal_params": signal_params,
        "backtest_params": backtest_params,
        "indicator_params2": indicator_params2,
        "indicator_enabled2": indicator_enabled2,
        "mapping_data": mapping_data,
    }


def get_indicator_params(
    num: int = 1, update_params: dict = {}, dtype_dict=default_dtype_dict
) -> dict:
    """
    生成默认指标参数，并允许通过 update 字典进行覆盖。

    参数:
        num (int): 每个指标默认参数列表的长度。
        update (dict): 用于更新默认参数的字典。
                       键必须是已知的指标名称，值必须与默认参数的结构匹配。

    返回:
        dict: 包含所有指标参数的字典。

    Raises:
        AssertionError: 如果 update 字典的格式不符合预期。
    """
    default_template = {
        sma_name: sma_spec,
        sma2_name: sma2_spec,
        bbands_name: bbands_spec,
        atr_name: atr_spec,
    }
    assert indicator_count == len(default_template.keys()), (
        f"指标数量不匹配 {indicator_count} {len(default_template.keys())}"
    )

    id_history = set()
    for k, v in default_template.items():
        assert v["param_count"] == len(v["default_params"]), (
            f"默认参数数量不对 {v['param_count']} {len(v['default_params'])}"
        )
        assert v["id"] not in id_history, f"指标id不允许重复 {k}"
        id_history.add(v["id"])

    # 初始化默认参数
    default_params = {
        k: [v["default_params"] for i in range(num)]
        for k, v in default_template.items()
    }

    # 遍历 update 字典并进行验证和更新
    for key, value_list in update_params.items():
        assert key in default_params, f"key '{key}' 无法识别"

        # 验证参数列表的数量
        assert len(default_params[key]) == len(value_list), (
            f"指标 '{key}' 的参数列表数量不匹配: 预期 {len(default_params[key])}, 实际 {len(value_list)}"
        )

        # 验证每个参数项的长度
        for i, item in enumerate(value_list):
            assert len(default_params[key][i]) == len(item), (
                f"指标 '{key}' 的第 {i + 1} 组参数长度不匹配: 预期 {len(default_params[key][i])}, 实际 {len(item)}"
            )

        # 更新参数
        default_params[key] = value_list

    return tuple(
        ensure_c_contiguous(np.array(v, dtype=dtype_dict["np"]["float"]))
        for k, v in default_params.items()
    )


def get_indicator_enabled(update_params={}, dtype_dict=default_dtype_dict):
    params = np.zeros(indicator_count, dtype=dtype_dict["np"]["bool"])
    for k, v in update_params.items():
        params[k] = bool(v)
    return ensure_c_contiguous(params)


def get_signal_params(params=[], dtype_dict=default_dtype_dict):
    params = np.array(params, dtype=dtype_dict["np"]["int"])
    return ensure_c_contiguous(params)


def get_backtest_params(num=1, params=[], dtype_dict=default_dtype_dict):
    default_params = [[0] for i in range(num)]
    params = np.array(default_params, dtype=dtype_dict["np"]["float"])
    return ensure_c_contiguous(params)


def get_mapping_data(data1, data2, dtype_dict=default_dtype_dict):
    # todo 具体逻辑待完善
    params = np.array([1, 2, 3], dtype=dtype_dict["np"]["int"])
    return ensure_c_contiguous(params)


def perpare_data(path, data_size=None, dtype_dict=default_dtype_dict):
    df_data = load_tohlcv_from_csv(path, data_size, dtype_dict)
    np_data = convert_tohlcv_numpy(df_data, dtype_dict)

    return df_data, ensure_c_contiguous(np_data)
