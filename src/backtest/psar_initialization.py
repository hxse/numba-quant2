import numba as nb
import numpy as np
import math

from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from src.indicators.psar import (
    psar_init,
    PsarState,
)  # 导入 PsarState 类型和 psar_init 函数

from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper

dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


# 定义新函数的签名
# 返回值：(new_psar_state: PsarState, psar_display_price: float)
signature = nb.types.Tuple((PsarState, nb_float_type))(
    nb_int_type,  # current_idx (当前 K 线的索引 i)
    nb_float_type[:],  # high_arr
    nb_float_type[:],  # low_arr
    nb_float_type[:],  # close_arr
    nb_float_type,  # af0
    nb_int_type,  # force_direction_int (强制方向：1多头，-1空头)
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def initialize_psar_state(
    current_idx,
    full_high_arr,
    full_low_arr,
    full_close_arr,
    af0,
    force_direction_int,
):
    """
    封装 PSAR 初始化的逻辑，返回初始 PSAR 状态和用于显示的 PSAR 价格。

    参数:
        current_idx: 当前 K 线的索引 (i)。
        full_high_arr, full_low_arr, full_close_arr: 完整的 OHLCV 数组。
        af0: 初始加速因子。
        force_direction_int: 强制 PSAR 的初始方向 (1: 多头, -1: 空头)。

    返回:
        (new_psar_state: PsarState,    # 初始化后的 PSAR 状态
         psar_display_price: float)    # 用于绘图的 PSAR 值
    """
    # 确保 new_psar_state 匹配 PsarState 的类型，例如 (False, np.nan, np.nan, np.nan)
    # 如果 PsarState 是一个 numba.typed.Tuple，你需要这样初始化它
    new_psar_state = (
        False,
        np.nan,
        np.nan,
        np.nan,
    )  # 假设 PsarState 是 (bool, float, float, float)
    psar_display_price = np.nan

    # psar_init 需要至少两根 K 线数据 (current_idx 和 current_idx - 1)
    if current_idx >= 1:
        # 从完整数组中提取 psar_init 所需的标量数据
        high_prev = full_high_arr[current_idx - 1]
        high_curr = full_high_arr[current_idx]
        low_prev = full_low_arr[current_idx - 1]
        low_curr = full_low_arr[current_idx]
        close_prev = full_close_arr[
            current_idx - 1
        ]  # psar_init 中使用 close[0]，对应这里的前一根 K 线收盘价

        initial_state_tuple = psar_init(
            high_prev,
            high_curr,
            low_prev,
            low_curr,
            close_prev,
            af0,
            force_direction_int,
        )

        # 检查 psar_init 的结果是否有效
        # initial_state_tuple[1] 是 current_psar，检查其是否为 NaN
        if not math.isnan(initial_state_tuple[1]):
            new_psar_state = initial_state_tuple
            # 根据强制方向设置 psar_display_price
            # 注意：这里和 force_direction_int 相关的逻辑可能需要根据实际需求调整
            # 你的逻辑是：
            # 如果强制多头 (1)，并且初始状态确实是多头 (new_psar_state[0]为 True)，则显示 psar 值
            # 如果强制空头 (-1)，并且初始状态确实是空头 (new_psar_state[0]为 False)，则显示 psar 值
            if force_direction_int == 1:  # 强制多头
                psar_display_price = new_psar_state[1] if new_psar_state[0] else np.nan
            elif force_direction_int == -1:  # 强制空头
                psar_display_price = (
                    new_psar_state[1] if not new_psar_state[0] else np.nan
                )
            else:  # 如果不是强制方向 (force_direction_int == 0 或其他值)，直接使用计算出的PSAR值
                psar_display_price = new_psar_state[1]

    return new_psar_state, psar_display_price
