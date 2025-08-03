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
    nb_float_type[:],  # full_high_arr (完整的 K 线高点数组)
    nb_float_type[:],  # full_low_arr (完整的 K 线低点数组)
    nb_float_type[:],  # full_close_arr (完整的 K 线收盘价数组)
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
    new_psar_state = (False, np.nan, np.nan, np.nan)
    psar_display_price = np.nan

    # psar_init 需要至少两根 K 线数据 (current_idx 和 current_idx - 1)
    if current_idx >= 1:
        # psar_init 期望传入 K 线数据片段，且片段内部索引从0开始。
        # full_arr[current_idx-1 : current_idx+1] 正好提供了索引0(current_idx-1)和索引1(current_idx)的两根K线。
        initial_state_tuple = psar_init(
            full_high_arr[current_idx - 1 : current_idx + 1],
            full_low_arr[current_idx - 1 : current_idx + 1],
            full_close_arr[current_idx - 1 : current_idx + 1],
            af0,
            force_direction_int,
        )

        # 检查 psar_init 的结果是否有效
        if not math.isnan(initial_state_tuple[1]):
            new_psar_state = initial_state_tuple
            # 根据强制方向设置 psar_display_price
            if force_direction_int == 1:  # 强制多头
                psar_display_price = new_psar_state[1] if new_psar_state[0] else np.nan
            elif force_direction_int == -1:  # 强制空头
                psar_display_price = (
                    new_psar_state[1] if not new_psar_state[0] else np.nan
                )
        # 如果初始化失败，psar_state 和 psar_display_price 保持为 NaN，表示无效

    return new_psar_state, psar_display_price
