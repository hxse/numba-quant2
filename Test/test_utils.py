import numpy as np
from typing import Tuple

GLOBAL_RTOL = 1e-5
GLOBAL_ATOL = 1e-8


def _count_leading_nans(arr: np.ndarray) -> int:
    """
    计算 NumPy 数组开头连续 NaN 值的数量。

    Args:
        arr (np.ndarray): 输入的 NumPy 数组。

    Returns:
        int: 开头连续 NaN 值的数量。
    """
    if arr.size == 0:
        return 0

    # 找到所有非 NaN 值的索引
    non_nan_indices = np.where(~np.isnan(arr))[0]

    if non_nan_indices.size == 0:
        # 如果没有非 NaN 值，则所有元素都是 NaN
        return arr.size
    else:
        # 第一个非 NaN 值的索引就是前导 NaN 的数量
        return non_nan_indices[0]


def get_leading_nan_counts_for_two_arrays(
    arr1: np.ndarray, arr2: np.ndarray
) -> Tuple[int, int]:
    """
    计算两个 NumPy 数组开头 NaN 值的数量。

    Args:
        arr1 (np.ndarray): 第一个 NumPy 数组。
        arr2 (np.ndarray): 第二个 NumPy 数组。

    Returns:
        Tuple[int, int]: 一个元组，包含 arr1 和 arr2 开头 NaN 值的数量。
    """
    nan_count_arr1 = _count_leading_nans(arr1)
    nan_count_arr2 = _count_leading_nans(arr2)
    return nan_count_arr1, nan_count_arr2


def assert_indicator_same(array1, array2, indicator_name, params_str):
    """
    通用函数，用于比较 Pandas TA 和 NumPy 实现的指标结果。
    """
    print("\n")
    # 如果是布尔数组，直接使用 assert_array_equal，并跳过 NaN 检查和 max_diff 计算

    if array1.dtype == bool or array2.dtype == bool:
        np.testing.assert_array_equal(
            array1,
            array2,
            err_msg=f"{indicator_name} calculation mismatch for {params_str}",
        )
    else:
        valid_indices = ~np.isnan(array1) & ~np.isnan(array2)

        array1_nan_count, array2_nan_count = get_leading_nan_counts_for_two_arrays(
            array1, array2
        )
        print(
            f"{indicator_name} ({params_str}) array1_nan_count: {array1_nan_count} (type: {type(array1).__name__}) array2_nan_count: {array2_nan_count} (type: {type(array2).__name__})"
        )
        assert array1_nan_count == array2_nan_count, (
            f"{indicator_name} leading NaN count mismatch: array1 has {array1_nan_count}, array2 has {array2_nan_count}"
        )

        # 计算并打印最大差值，只考虑有效索引
        max_diff = (
            np.max(np.abs(array1[valid_indices] - array2[valid_indices]))
            if np.any(valid_indices)
            else 0.0
        )
        print(f"{indicator_name} ({params_str}) - Max difference: {max_diff:.4e}")

        np.testing.assert_allclose(
            array1[valid_indices],
            array2[valid_indices],
            rtol=GLOBAL_RTOL,
            atol=GLOBAL_ATOL,
            err_msg=f"{indicator_name} calculation mismatch for {params_str}",
        )
    print(f"{indicator_name} ({params_str}) accuracy test passed.")


def assert_indicator_different(array1, array2, indicator_name, params_str):
    """
    检测两个指标的结果是否不同。
    如果 assert_indicator_same 抛出异常（表明结果不同），则表示这个差异性测试通过。
    如果 assert_indicator_same 成功运行（无异常，表明结果相同），则表示这个差异性测试失败。
    """
    print(f"\n--- {indicator_name} ({params_str}) - 差异性测试 (期望不同) ---")
    try:
        assert_indicator_same(array1, array2, indicator_name, params_str)
        raise AssertionError(
            f"❌ {indicator_name} ({params_str}) array1 和 array2 被判断为相同，但测试期望是不同，测试失败！"
        )
    except AssertionError as e:
        if f"array1 和 array2 被判断为相同" in str(e):
            raise e
        else:
            print(
                f"  ✅ {indicator_name} ({params_str}) array1 和 array2 被判断为不同（符合期望）。"
            )
            print(f"    详细信息: {e}")  # 打印 assert_indicator_same 报告的差异细节
