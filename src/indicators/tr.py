import numba as nb
import numpy as np


# from enum import Enum


# tr_id = 3
# tr_name = "tr"

# tr_spec = {
#     "id": tr_id,
#     "name": tr_name,
#     "ori_name": tr_name,
#     "result_name": ["tr"],
#     "default_params": [],
#     "param_count": 0,
#     "result_count": 1,
#     "temp_count": 0,
# }

from utils.numba_params import nb_params
from utils.data_types import get_numba_data_types
from utils.numba_utils import nb_wrapper


dtype_dict = get_numba_data_types(nb_params.get("enable64", True))
nb_int_type = dtype_dict["nb"]["int"]
nb_float_type = dtype_dict["nb"]["float"]
nb_bool_type = dtype_dict["nb"]["bool"]


signature = nb.void(
    nb_float_type[:], nb_float_type[:], nb_float_type[:], nb_float_type[:]
)


@nb_wrapper(
    mode=nb_params["mode"],
    signature=signature,
    cache_enabled=nb_params.get("cache", True),
)
def calculate_tr(high, low, close, tr_result):
    n = len(high)
    # 对于第一个数据点，prev_close 不存在，TR 应该为 np.nan
    if n > 0:
        tr_result[0] = np.nan  # 保持不变

    for i in range(1, n):
        # 检查输入值是否为 np.nan，如果任一为 NaN，则结果为 NaN
        # 在 Numba 中，对于浮点数 NaN 的检查，更常用的是 x != x
        if high[i] != high[i] or low[i] != low[i] or close[i - 1] != close[i - 1]:
            tr_result[i] = np.nan
            continue

        # 计算 TR
        range1 = high[i] - low[i]
        range2 = abs(high[i] - close[i - 1])
        range3 = abs(low[i] - close[i - 1])
        tr_result[i] = max(range1, range2, range3)
