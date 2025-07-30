import numba as nb
import numpy as np

# --- 1. 模拟数据类型和签名 ---
# 简化 Numba 类型，以便复现核心问题
nb_int_type = nb.int64
nb_float_type = nb.float64

# 简化类型用于最小示例
# 假设 _unpack_flatten 返回 7 个元素，其中一个就是导致问题的 UniTuple
# 例如：(array, UniTuple, array, array, array, array, array)
# 简化为 (array, UniTuple(array x 4), array, array, array, array, array)
# 关键在于模拟 `UniTuple(array(float64, 2d, A) x 4)` 作为某个参数
Array2D = nb.float64[:, :]
Array3D = nb.float64[:, :, :]

# 模拟那个 UniTuple(array(float64, 2d, A) x 4)
# Numba 内部有时会把这种 UniTuple 当作“标量”处理
UniTupleOfArrays4x2D = nb.types.UniTuple(Array2D, 4)

# 模拟 _unpack_flatten 的返回类型：7个元素
FlattenedParamsType = nb.types.Tuple(
    (
        Array2D,  # data_args (简化)
        UniTupleOfArrays4x2D,  # indicator_params (这里是问题所在)
        Array2D,  # indicator_params2 (简化)
        Array3D,  # indicator_result (简化)
        Array3D,  # indicator_result2 (简化)
        Array2D,  # signal_args (简化)
        Array2D,  # backtest_args (简化)
    )
)

# 模拟 _unpack_flatten_idx 的返回类型：7个子参数
# 注意，这里面会把上面的 UniTupleOfArrays4x2D 再次拆开
FlattenedChildParamsType = nb.types.Tuple(
    (
        nb.float64[:],  # _data_args (简化为1D，因为通常是取一行)
        nb.float64[:],  # indicator_params_child (扁平化，从 UniTuple中取一个)
        nb.float64[:],  # indicator_params2_child
        nb.float64[:, :],  # indicator_result_child
        nb.float64[:, :],  # indicator_result2_child
        nb.float64[:],  # signal_args_child
        nb.float64[:],  # backtest_args_child
    )
)

# --- 2. 模拟辅助函数 ---


@nb.njit(
    FlattenedParamsType(nb.typeof(None)), cache=True
)  # None只是占位，实际应该接收params的类型
def _unpack_flatten_simulated(params):
    # 模拟返回一个包含 UniTuple 的元组
    # 注意：这里的返回类型要和 FlattenedParamsType 严格匹配
    arr1 = np.ones((100, 2), dtype=np.float64)  # 增加行数以支持idx索引
    arr2_ut = (
        np.ones((100, 5), dtype=np.float64),
        np.ones((100, 6), dtype=np.float64),
        np.ones((100, 7), dtype=np.float64),
        np.ones((100, 8), dtype=np.float64),
    )
    arr3 = np.ones((100, 3), dtype=np.float64)
    arr4 = np.ones((100, 4, 2), dtype=np.float64)
    arr5 = np.ones((100, 5, 3), dtype=np.float64)
    arr6 = np.ones((100, 9), dtype=np.float64)
    arr7 = np.ones((100, 10), dtype=np.float64)
    return (arr1, arr2_ut, arr3, arr4, arr5, arr6, arr7)


# 模拟 _unpack_flatten_idx，接收上面扁平化的7个参数，再加一个idx
@nb.njit(
    FlattenedChildParamsType(
        Array2D,  # data_args
        UniTupleOfArrays4x2D,  # indicator_params
        Array2D,  # indicator_params2
        Array3D,  # indicator_result
        Array3D,  # indicator_result2
        Array2D,  # signal_args
        Array2D,  # backtest_args
        nb_int_type,  # idx
    ),
    cache=True,
)
def _unpack_flatten_idx_simulated(
    data_args,
    indicator_params,
    indicator_params2,
    indicator_result,
    indicator_result2,
    signal_args,
    backtest_args,
    idx,
):
    # 模拟从这些参数中取出索引为 idx 的子元素
    _data_args = data_args[idx, :]
    # 这里是关键，从 UniTuple 中取出一个数组的切片
    # Numba期望这里能正确推断出是1D数组
    _indicator_params_child = indicator_params[0][
        idx, :
    ]  # 假设取 UniTuple 的第一个元素并切片
    _indicator_params2_child = indicator_params2[idx, :]
    _indicator_result_child = indicator_result[idx, :, :]
    _indicator_result2_child = indicator_result2[idx, :, :]
    _signal_args_child = signal_args[idx, :]
    _backtest_args_child = backtest_args[idx, :]

    return (
        _data_args,
        _indicator_params_child,
        _indicator_params2_child,
        _indicator_result_child,
        _indicator_result2_child,
        _signal_args_child,
        _backtest_args_child,
    )


# 模拟 _parallel_calc，接收重组后的子参数
@nb.njit(
    nb.void(
        nb.types.Tuple(
            (
                nb.float64[:],
                nb.types.Tuple(
                    (nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64[:, :])
                ),
                nb.float64[:],
                nb.float64[:],
            )
        )
    ),
    cache=True,
)
def _parallel_calc_simulated(params_child):
    # 实际计算逻辑，这里简化为打印
    _data_args, indicator_args_child, _signal_args_child, _backtest_args_child = (
        params_child
    )
    # Numba不允许在jit模式下直接print numpy数组，所以这里不做复杂操作
    # print(f"Processing idx, first data element: {_data_args[0]}") # 示例打印
    pass  # 实际会进行一些计算


# --- 3. 核心测试函数 ---


def run_test_case(test_mode):
    print(f"\n--- Running Test Case: {test_mode} ---")

    # 模拟 Numba 编译时获取这些函数
    _get_conf_count = nb.njit(nb.int64(nb.typeof(None)), cache=True)(
        lambda p: 100
    )  # 模拟返回一个固定数量
    _unpack_flatten = _unpack_flatten_simulated
    _unpack_flatten_idx = _unpack_flatten_idx_simulated
    _parallel_calc = _parallel_calc_simulated

    params = None  # 模拟一个参数，实际会被_unpack_flatten处理

    conf_count = _get_conf_count(params)

    res = _unpack_flatten(
        params
    )  # res 是一个包含 7 个元素的元组，其中 res[1] 是 UniTupleOfArrays4x2D
    (
        data_args,
        indicator_params,
        indicator_params2,
        indicator_result,
        indicator_result2,
        signal_args,
        backtest_args,
    ) = res

    # Numba JIT 编译的内部循环函数
    # 这个函数接收一个 `FlattenedParamsType` 类型的参数 (res_inner)
    @nb.njit(nb.void(FlattenedParamsType), parallel=True)
    def _cpu_parallel_calc_jit_inner(res_inner):
        # 内部解包res_inner，这些变量现在是 Numba 类型
        (
            _data_args_inner,
            _indicator_params_inner,
            _indicator_params2_inner,
            _indicator_result_inner,
            _indicator_result2_inner,
            _signal_args_inner,
            _backtest_args_inner,
        ) = res_inner

        for idx in nb.prange(conf_count):
            _res_to_pass_to_unpack_idx = None  # 初始化

            if test_mode == "Explicit Variables (Works)":
                # 方式 A: 显式列出变量 (成功)
                # _unpack_flatten_idx 期望接收 8 个独立参数
                (
                    _data_args,
                    indicator_params_child,
                    indicator_params2_child,
                    indicator_result_child,
                    indicator_result2_child,
                    signal_args_child,
                    backtest_args_child,
                ) = _unpack_flatten_idx(
                    _data_args_inner,
                    _indicator_params_inner,
                    _indicator_params2_inner,
                    _indicator_result_inner,
                    _indicator_result2_inner,
                    _signal_args_inner,
                    _backtest_args_inner,
                    idx,
                )

            elif test_mode == "Indexed `res` (Fails)":
                # 方式 B: 使用 res[i] 索引 (失败)
                # _unpack_flatten_idx 期望接收 8 个独立参数。
                # res_inner[0] 到 res_inner[6] 是 7 个参数，idx 是第 8 个参数。
                # 这里的 Numba 错误仍然是由于它把 res_inner[1] (UniTupleOfArrays4x2D) 误认为是标量
                # 当作为独立参数传入时，它无法正确解构
                (
                    _data_args,
                    indicator_params_child,
                    indicator_params2_child,
                    indicator_result_child,
                    indicator_result2_child,
                    signal_args_child,
                    backtest_args_child,
                ) = _unpack_flatten_idx(
                    res_inner[0],
                    res_inner[1],
                    res_inner[2],
                    res_inner[3],
                    res_inner[4],
                    res_inner[5],
                    res_inner[6],
                    idx,
                )

            elif test_mode == "Starred `res` (Fails, initial problem)":
                # 方式 C: 使用 *res (最初的问题，也会失败)
                # 这里我们模拟 `_unpack_flatten_idx(*res_inner, idx)` 的调用方式
                # _unpack_flatten_idx 期望 8 个独立参数，而 *res_inner 会展开成 7 个参数
                # 加上 idx 刚好 8 个，但关键是 Numba 对 *res_inner 的类型推断问题
                (
                    _data_args,
                    indicator_params_child,
                    indicator_params2_child,
                    indicator_result_child,
                    indicator_result2_child,
                    signal_args_child,
                    backtest_args_child,
                ) = _unpack_flatten_idx(*res_inner, idx)
            else:
                raise ValueError("Invalid test mode")

            # 后续的参数重组和调用保持不变
            indicator_args_child = (
                indicator_params_child,
                indicator_params2_child,
                indicator_result_child,
                indicator_result2_child,
            )
            _params_child = (
                _data_args,
                indicator_args_child,
                signal_args_child,
                backtest_args_child,
            )
            _parallel_calc(_params_child)

    try:
        # 调用 Numba 编译后的内部函数
        # 注意：这里传入的是最开始解包后的 'res'，因为 _cpu_parallel_calc_jit_inner 预期接收的是一个 FlattenedParamsType
        _cpu_parallel_calc_jit_inner(res)
        print(f"Test Case '{test_mode}' SUCCEEDED.")
    except Exception as e:
        print(f"Test Case '{test_mode}' FAILED with error: {type(e).__name__}: {e}")


# --- 运行测试案例 ---
# 确保你的 Numba 版本能够复现问题
# 如果 Numba 足够智能，某些Fails的案例可能在你的环境中通过

# run_test_case("Explicit Variables (Works)")
# run_test_case("Indexed `res` (Fails)")  # 理论上会失败
run_test_case("Starred `res` (Fails, initial problem)")  # 理论上会失败
