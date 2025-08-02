import numba as nb
import numpy as np
from .tr import calculate_tr
from .rma import calculate_rma
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


signature = nb.void(
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
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calculate_psar(
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

    # --- 初始状态设置，模拟 pandas_ta 的内部逻辑 ---
    # 确定初始趋势：通过比较第一和第二根K线的DM来模拟 pandas_ta 的 _falling 函数
    up_dm = high[1] - high[0]
    dn_dm = low[0] - low[1]

    # 如果下跌动量大于上涨动量且大于0，则初始趋势为下跌
    is_falling_initial = False
    if dn_dm > up_dm and dn_dm > 0:
        is_falling_initial = True

    # is_long 与 is_falling_initial 相反
    is_long = not is_falling_initial

    # 初始化当前 PSAR 值。如果提供了 close，则使用 close[0]，否则根据趋势使用 high[0] 或 low[0]。
    # 我们假设 close 总是提供，因为 Pandas TA 在 close != None 时优先使用 close[0]。
    current_psar = close[0]

    # 初始化极端点 (EP)
    current_ep = low[0] if is_falling_initial else high[0]

    # 初始化加速因子 (AF)
    current_af = af0

    # 初始化结果数组的第0个索引以匹配 Pandas TA
    # Pandas TA 将 _af[0] 设置为 af0，reversal[0] 设置为 0。
    psar_af_result[0] = current_af
    psar_reversal_result[0] = 0.0  # 初始时刻没有反转

    # --- 主要计算循环 ---
    # 循环从第二个数据点 (索引 1) 开始
    for i in range(1, n):
        prev_psar = current_psar
        prev_ep = current_ep
        prev_af = current_af

        # 1. 计算下一根 K 线的原始 PSAR 候选值 (未经穿透调整)
        if is_long:
            next_psar_raw_candidate = prev_psar + prev_af * (prev_ep - prev_psar)
        else:  # is_short
            next_psar_raw_candidate = prev_psar - prev_af * (prev_psar - prev_ep)

        # 2. 判断是否发生反转：使用原始 PSAR 候选值与当前 K 线的价格进行比较
        # 这一步是关键，它与 Pandas TA 的行为保持一致
        reversal = False
        if is_long:  # 如果当前趋势为上涨 (Long)
            # 如果当前 K 线的最低价跌破了原始 PSAR 候选值，则发生反转
            if low[i] < next_psar_raw_candidate:
                reversal = True
        else:  # 如果当前趋势为下跌 (Short)
            # 如果当前 K 线的最高价突破了原始 PSAR 候选值，则发生反转
            if high[i] > next_psar_raw_candidate:
                reversal = True

        # 3. 对 PSAR 进行穿透检查 (SAR 不能穿透前一根 K 线的价格)
        # 得到经过穿透调整的当前 PSAR 值
        if is_long:  # 当前趋势为上涨
            # PSAR 不能高于前一根 K 线的最低价
            current_psar = min(next_psar_raw_candidate, low[i - 1])
        else:  # 当前趋势为下跌
            # PSAR 不能低于前一根 K 线的最高价
            current_psar = max(next_psar_raw_candidate, high[i - 1])

        # 4. 更新极端点 (EP) 和加速因子 (AF)
        # 这些更新发生在反转检查（但不是反转处理）之前，以更新 EP/AF 的增长
        if is_long:
            if high[i] > current_ep:  # 如果当前 K 线创下新高点
                current_ep = high[i]
                current_af = min(max_af, prev_af + af_step)  # 增加 AF，但不超过 max_af
        else:  # is_short
            if low[i] < current_ep:  # 如果当前 K 线创下新低点
                current_ep = low[i]
                current_af = min(max_af, prev_af + af_step)  # 增加 AF，但不超过 max_af

        # 5. 处理反转（如果发生）
        if reversal:
            is_long = not is_long  # 反转趋势
            current_af = af0  # 重置加速因子为初始值

            # 反转后的新 PSAR 值是上一个极端点 (prev_ep)
            current_psar = prev_ep

            # 确保反转后的新 PSAR 不会穿透当前 K 线（这是反转后的额外修正）
            if is_long:  # 现在处于上涨趋势
                if (
                    current_psar > low[i]
                ):  # 如果新 SAR 高于当前 K 线最低价，则设为最低价
                    current_psar = low[i]
                current_ep = high[i]  # 新 EP 是当前 K 线的最高价
            else:  # 现在处于下跌趋势
                if (
                    current_psar < high[i]
                ):  # 如果新 SAR 低于当前 K 线最高价，则设为最高价
                    current_psar = high[i]
                current_ep = low[i]  # 新 EP 是当前 K 线的最低价

        # 6. 填充结果数组
        psar_af_result[i] = current_af
        psar_reversal_result[i] = float(int(reversal))  # 确保结果为浮点数 (0.0 或 1.0)

        # 根据最终确定的趋势填充 Long 或 Short PSAR
        if is_long:
            psar_long_result[i] = current_psar
            psar_short_result[i] = np.nan  # 如果是 Long 趋势，Short PSAR 为 NaN
        else:
            psar_short_result[i] = current_psar
            psar_long_result[i] = np.nan  # 如果是 Short 趋势，Long PSAR 为 NaN


signature = nb.void(
    *get_indicator_wrapper_signal(nb_int_type, nb_float_type, nb_bool_type)
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calculate_psar_wrapper(
    tohlcv, indicator_params_child, indicator_result_child, float_temp_array_child, _id
):
    time = tohlcv[:, 0]
    open = tohlcv[:, 1]
    high = tohlcv[:, 2]
    low = tohlcv[:, 3]
    close = tohlcv[:, 4]
    volume = tohlcv[:, 5]

    psar_indicator_params_child = indicator_params_child[_id]
    psar_indicator_result_child = indicator_result_child[_id]

    af0 = psar_indicator_params_child[0]
    af_step = psar_indicator_params_child[1]
    max_af = psar_indicator_params_child[2]

    psar_long_result = psar_indicator_result_child[:, 0]
    psar_short_result = psar_indicator_result_child[:, 1]
    psar_af_result = psar_indicator_result_child[:, 2]
    psar_reversal_result = psar_indicator_result_child[:, 3]

    # psar_period 不用显示转换类型, numba会隐式把小数截断成整数(小数部分丢弃)
    calculate_psar(
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
