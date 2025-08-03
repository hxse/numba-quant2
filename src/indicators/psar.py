import numba as nb
import numpy as np

from utils.data_types import (
    get_indicator_params_child,
    get_indicator_result_child,
    get_indicator_wrapper_signal,
)
from .indicators_tool import check_bounds

from enum import Enum

psar_id = 4
psar_name = "psar"

psar_spec = {
    "id": psar_id,
    "name": psar_name,
    "ori_name": psar_name,
    "result_name": [
        "psar_long_result",
        "psar_short_result",
        "psar_af_result",
        "psar_reversal_result",
    ],
    "default_params": [0.02, 0.02, 0.2],
    "param_count": 3,
    "result_count": 4,
    "temp_count": 0,
}

from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]

# --- PSAR 状态元组定义 ---
PsarState = nb.types.Tuple((nb_bool_type, nb_float_type, nb_float_type, nb_float_type))

# --- PSAR 初始化函数 ---
signature_init = PsarState(
    nb_float_type[:],  # high (至少需要前两根 K 线)
    nb_float_type[:],  # low (至少需要前两根 K 线)
    nb_float_type[:],  # close (至少需要前两根 K 线)
    nb_float_type,  # af0
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature_init,
    cache_enabled=nb_params.get("cache", True),
)
def psar_init(high, low, close, af0):
    """
    初始化 PSAR 算法的初始状态。
    需要至少两根 K 线数据来确定初始趋势。
    返回一个元组：(is_long, current_psar, current_ep, current_af)
    """
    if len(close) < 2:
        return (False, np.nan, np.nan, np.nan)

    # 确定初始趋势，与 pandas_ta 的 _falling 函数一致
    up_dm = high[1] - high[0]
    dn_dm = low[0] - low[1]
    is_falling_initial = dn_dm > up_dm and dn_dm > 0
    is_long = not is_falling_initial

    # 初始化 PSAR 值，直接使用 close[0]，与 pandas_ta 的 sar[0] = close[0] 一致
    current_psar = close[0]

    # 初始化极端点 (EP)
    current_ep = low[0] if is_falling_initial else high[0]

    # 初始化加速因子 (AF)
    current_af = af0

    return (is_long, current_psar, current_ep, current_af)


# --- PSAR 实时更新函数 ---
signature_update = nb.types.Tuple(
    (
        nb_bool_type,
        nb_float_type,
        nb_float_type,
        nb_float_type,
        nb_float_type,
        nb_float_type,
        nb_float_type,
    )
)(
    PsarState,  # prev_state
    nb_float_type,  # current_high
    nb_float_type,  # current_low
    nb_float_type,  # prev_high
    nb_float_type,  # prev_low
    nb_float_type,  # af_step
    nb_float_type,  # max_af
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature_update,
    cache_enabled=nb_params.get("cache", True),
)
def psar_update(
    prev_state, current_high, current_low, prev_high, prev_low, af_step, max_af
):
    """
    根据前一根 K 线后的 PSAR 状态和当前 K 线的数据，计算新的 PSAR 值并更新状态。
    返回一个元组：(new_is_long, new_psar, new_ep, new_af, psar_long_val, psar_short_val, reversal)
    """
    prev_is_long, prev_psar, prev_ep, prev_af = prev_state

    # 1. 计算下一根 K 线的原始 PSAR 候选值
    if prev_is_long:
        next_psar_raw_candidate = prev_psar + prev_af * (prev_ep - prev_psar)
    else:
        next_psar_raw_candidate = prev_psar - prev_af * (prev_psar - prev_ep)

    # 2. 判断是否发生反转
    reversal = False
    if prev_is_long:
        if current_low < next_psar_raw_candidate:
            reversal = True
    else:
        if current_high > next_psar_raw_candidate:
            reversal = True

    # 3. 对 PSAR 进行穿透检查
    current_psar = 0.0
    if prev_is_long:
        current_psar = min(next_psar_raw_candidate, prev_low)
    else:
        current_psar = max(next_psar_raw_candidate, prev_high)

    # 4. 更新极端点 (EP) 和加速因子 (AF)
    new_ep = prev_ep
    new_af = prev_af
    if prev_is_long:
        if current_high > new_ep:
            new_ep = current_high
            new_af = min(max_af, prev_af + af_step)
    else:
        if current_low < new_ep:
            new_ep = current_low
            new_af = min(max_af, prev_af + af_step)

    # 5. 处理反转（如果发生）
    new_is_long = prev_is_long
    if reversal:
        new_is_long = not prev_is_long
        new_af = af_step
        current_psar = prev_ep
        if new_is_long:
            if current_psar > current_low:
                current_psar = current_low
            new_ep = current_high
        else:
            if current_psar < current_high:
                current_psar = current_high
            new_ep = current_low

    # 6. 确定返回的 PSAR 值
    psar_long_val = np.nan
    psar_short_val = np.nan
    if new_is_long:
        psar_long_val = current_psar
    else:
        psar_short_val = current_psar

    return (
        new_is_long,
        current_psar,
        new_ep,
        new_af,
        psar_long_val,
        psar_short_val,
        float(int(reversal)),
    )


# --- calculate_psar_all ---
signature_all = nb.void(
    nb_float_type[:],
    nb_float_type[:],
    nb_float_type[:],
    nb_float_type,
    nb_float_type,
    nb_float_type,
    nb_float_type[:],
    nb_float_type[:],
    nb_float_type[:],
    nb_float_type[:],
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature_all,
    cache_enabled=nb_params.get("cache", True),
)
def calculate_psar_all(
    high,
    low,
    close,
    af0,
    af_step,
    max_af,
    psar_long_result,
    psar_short_result,
    psar_af_result,
    psar_reversal_result,
):
    # 越界检查
    if check_bounds(close, 1, psar_long_result) == 0:
        return

    n = len(close)

    # 初始化结果数组为 NaN
    psar_long_result[:] = np.nan
    psar_short_result[:] = np.nan
    psar_af_result[:] = np.nan
    psar_reversal_result[:] = np.nan

    if n < 2:
        return

    # 初始化索引 0 的 af 和 reversal，与 pandas_ta 一致
    psar_af_result[0] = af0
    psar_reversal_result[0] = 0.0

    # 使用 psar_init 获取初始状态
    initial_state = psar_init(high[:2], low[:2], close[:2], af0)
    is_long, current_psar, current_ep, current_af = initial_state

    if np.isnan(current_psar):
        return

    # 计算索引 1 的 PSAR，模仿 pandas_ta 的第一次迭代
    next_psar_raw_candidate = (
        current_psar + current_af * (current_ep - current_psar)
        if is_long
        else current_psar - current_af * (current_psar - current_ep)
    )
    current_psar = (
        min(next_psar_raw_candidate, low[0])
        if is_long
        else max(next_psar_raw_candidate, high[0])
    )

    # 检查反转
    reversal = (
        low[1] < next_psar_raw_candidate
        if is_long
        else high[1] > next_psar_raw_candidate
    )

    # 更新 EP 和 AF
    if is_long:
        if high[1] > current_ep:
            current_ep = high[1]
            current_af = min(max_af, current_af + af_step)
    else:
        if low[1] < current_ep:
            current_ep = low[1]
            current_af = min(max_af, current_af + af_step)

    # 处理反转
    if reversal:
        is_long = not is_long
        current_af = af0
        current_psar = current_ep
        if is_long:
            if current_psar > low[1]:
                current_psar = low[1]
            current_ep = high[1]
        else:
            if current_psar < high[1]:
                current_psar = high[1]
            current_ep = low[1]

    # 填充索引 1 的结果
    psar_long_result[1] = current_psar if is_long else np.nan
    psar_short_result[1] = current_psar if not is_long else np.nan
    psar_af_result[1] = current_af
    psar_reversal_result[1] = float(int(reversal))

    # 核心循环：从索引 2 开始
    for i in range(2, n):
        prev_state_tuple = (is_long, current_psar, current_ep, current_af)
        (
            new_is_long,
            new_psar,
            new_ep,
            new_af,
            psar_long_val,
            psar_short_val,
            reversal_val,
        ) = psar_update(
            prev_state_tuple, high[i], low[i], high[i - 1], low[i - 1], af_step, max_af
        )

        is_long = new_is_long
        current_psar = new_psar
        current_ep = new_ep
        current_af = new_af

        psar_long_result[i] = psar_long_val
        psar_short_result[i] = psar_short_val
        psar_af_result[i] = current_af
        psar_reversal_result[i] = reversal_val


# --- calculate_psar_wrapper ---
signature_wrapper = nb.void(
    *get_indicator_wrapper_signal(nb_int_type, nb_float_type, nb_bool_type)
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature_wrapper,
    cache_enabled=nb_params.get("cache", True),
)
def calculate_psar_wrapper(
    tohlcv, indicator_params_child, indicator_result_child, float_temp_array_child, _id
):
    high = tohlcv[:, 2]
    low = tohlcv[:, 3]
    close = tohlcv[:, 4]

    psar_indicator_params_child = indicator_params_child[_id]
    psar_indicator_result_child = indicator_result_child[_id]

    af0 = psar_indicator_params_child[0]
    af_step = psar_indicator_params_child[1]
    max_af = psar_indicator_params_child[2]

    psar_long_result = psar_indicator_result_child[:, 0]
    psar_short_result = psar_indicator_result_child[:, 1]
    psar_af_result = psar_indicator_result_child[:, 2]
    psar_reversal_result = psar_indicator_result_child[:, 3]

    calculate_psar_all(
        high,
        low,
        close,
        af0,
        af_step,
        max_af,
        psar_long_result,
        psar_short_result,
        psar_af_result,
        psar_reversal_result,
    )
