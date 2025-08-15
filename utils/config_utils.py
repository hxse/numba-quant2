from utils.data_loading import load_tohlcv_from_csv, convert_tohlcv_numpy
from utils.data_types import get_numba_data_types
import numpy as np

from src.indicators.indicators_wrapper import (
    IndicatorsId,
    indicators_id_array,
    indicators_spec,
)
from src.backtest.calculate_backtest import default_backtest_params

from src.signal.simple_template import simple_name, simple_spec

indicator_count = len(indicators_id_array)  # 最大的指标id值+1

default_signal_template = {
    simple_name: simple_spec,
}
default_signal_name = simple_name


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
    signal_name=default_signal_name,
    backtest_params={},
    dtype_dict=default_dtype_dict,
):
    signal_params, dependency, dependency2, exit_control = get_signal_params(
        signal_name=signal_name, dtype_dict=dtype_dict
    )
    indicator_enabled = {**indicator_enabled, **dependency}
    indicator_enabled2 = {**indicator_enabled2, **dependency2}
    backtest_params = {**backtest_params, **exit_control}

    indicator_params = get_indicator_params(
        num, update_params=indicator_update, dtype_dict=dtype_dict
    )
    indicator_enabled = get_indicator_enabled(
        update_params=indicator_enabled, dtype_dict=dtype_dict
    )
    indicator_name = get_indicator_name(
        indicator_enabled=indicator_enabled, dtype_dict=dtype_dict
    )
    indicator_col_name = get_indicator_col_name(
        indicator_enabled=indicator_enabled, dtype_dict=dtype_dict
    )

    backtest_params = get_backtest_params(
        num, params=backtest_params, dtype_dict=dtype_dict
    )

    indicator_params2 = get_indicator_params(
        num, update_params=indicator_update2, dtype_dict=dtype_dict
    )
    indicator_enabled2 = get_indicator_enabled(
        update_params=indicator_enabled2, dtype_dict=dtype_dict
    )
    indicator_name2 = get_indicator_name(
        indicator_enabled=indicator_enabled2, dtype_dict=dtype_dict
    )
    indicator_col_name2 = get_indicator_col_name(
        indicator_enabled=indicator_enabled2, dtype_dict=dtype_dict
    )

    # todo 待完善
    mapping_data = get_mapping_data([], [], dtype_dict=dtype_dict)

    return {
        "indicator_params": indicator_params,
        "indicator_name": indicator_name,
        "indicator_col_name": indicator_col_name,
        "indicator_enabled": indicator_enabled,
        "signal_params": signal_params,
        "backtest_params": backtest_params,
        "indicator_params2": indicator_params2,
        "indicator_name2": indicator_name2,
        "indicator_col_name2": indicator_col_name2,
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
    assert indicator_count == len(indicators_spec.keys()), (
        f"指标数量不匹配 {indicator_count} {len(indicators_spec.keys())}"
    )

    id_history = set()
    for k, v in indicators_spec.items():
        assert v["param_count"] == len(v["default_params"]), (
            f"默认参数数量不对 {v['param_count']} {len(v['default_params'])}"
        )
        assert v["id"] not in id_history, f"指标id不允许重复 {k}"
        id_history.add(v["id"])

    # 初始化默认参数
    default_params = {
        k: [v["default_params"] for i in range(num)] for k, v in indicators_spec.items()
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
        params[indicators_spec[k]["id"]] = bool(v)
    return ensure_c_contiguous(params)


def get_indicator_name(indicator_enabled=[], dtype_dict=default_dtype_dict):
    indicator_name = []
    for k, v in indicators_spec.items():
        if indicator_enabled[v["id"]]:
            indicator_name.append(v["name"])
    return indicator_name


def get_indicator_col_name(indicator_enabled=[], dtype_dict=default_dtype_dict):
    indicator_col_name = []
    for k, v in indicators_spec.items():
        if indicator_enabled[v["id"]]:
            indicator_col_name.append(v["result_name"])
    return indicator_col_name


def get_signal_params(signal_name="", dtype_dict=default_dtype_dict):
    if signal_name in default_signal_template:
        v = default_signal_template[signal_name]
        params = np.array([v["id"]], dtype=dtype_dict["np"]["int"])
        return (
            ensure_c_contiguous(params),
            v["dependency"],
            v["dependency2"],
            v["exit_control"],
        )
    else:
        raise RuntimeError(f"检测不到合法的signal name: {signal_name}")


def get_backtest_params(num=1, params={}, dtype_dict=default_dtype_dict):
    check_keys_exist(default_backtest_params, params)
    res = []
    for k, v in default_backtest_params.items():
        if k in params:
            res.append(params[k])
        else:
            res.append(v)

    default_params = [res for i in range(num)]
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


def check_keys_exist(base_dict, new_dict):
    """
    检查 new_dict 中的所有键是否都存在于 base_dict 中。

    参数:
    - base_dict (dict): 作为基准的字典，包含所有允许的键。
    - new_dict (dict): 需要被检查的字典。

    如果 new_dict 存在不属于 base_dict 的键，将抛出 ValueError。
    """
    # 将字典的键转换为集合，以便进行高效的差集操作
    base_keys = set(base_dict.keys())
    new_keys = set(new_dict.keys())

    # 找到 new_keys 中存在但 base_keys 中不存在的键
    # 差集操作 (new_keys - base_keys) 会返回这些多余的键
    extra_keys = new_keys - base_keys

    # 如果 extra_keys 集合不为空，说明存在多余的键
    if extra_keys:
        raise ValueError(
            f"提供的字典包含不合法的键: {', '.join(sorted(list(extra_keys)))}"
        )
