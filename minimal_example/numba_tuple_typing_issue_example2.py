import numba as nb
import numpy as np

# --- 1. 模拟数据类型和签名 ---
nb_int_type = nb.int64
nb_float_type = nb.float64

Array2D = nb.float64[:, :]
Array3D = nb.float64[:, :, :]

# 模拟那个 UniTuple(array(float64, 2d, A) x 4)
UniTupleOfArrays4x2D = nb.types.UniTuple(Array2D, 4)

# _unpack_flatten 的返回类型：7个元素
FlattenedParamsType = nb.types.Tuple(
    (
        Array2D,  # data_args (0)
        UniTupleOfArrays4x2D,  # indicator_params (1) - 这里是问题所在
        Array2D,  # indicator_params2 (2)
        Array3D,  # indicator_result (3)
        Array3D,  # indicator_result2 (4)
        Array2D,  # signal_args (5)
        Array2D,  # backtest_args (6)
    )
)

# _unpack_flatten_idx 的返回类型：7个子参数
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

# 定义一个新类型用于 _unpack_flatten_idx 接收的打包参数元组
FunctionInputPackedArgsType = nb.types.Tuple(
    (Array2D, UniTupleOfArrays4x2D, Array2D, Array3D, Array3D, Array2D, Array2D)
)

# --- 2. 模拟辅助函数 ---


@nb.njit(FlattenedParamsType(nb.typeof(None)), cache=True)
def _unpack_flatten_simulated(params):
    # 模拟返回一个包含 UniTuple 的元组
    arr1 = np.ones((100, 2), dtype=np.float64)
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


# 统一 _unpack_flatten_idx 函数，只接收一个打包元组和一个索引
@nb.njit(
    FlattenedChildParamsType(
        FunctionInputPackedArgsType,  # params元组的类型
        nb_int_type,  # idx的类型
    ),
    cache=True,
)
def _unpack_flatten_idx_simulated(params_tuple, idx):
    # 在函数内部解包或通过索引访问
    data_args = params_tuple[0]
    indicator_params = params_tuple[1]  # 这是 UniTupleOfArrays4x2D
    indicator_params2 = params_tuple[2]
    indicator_result = params_tuple[3]
    indicator_result2 = params_tuple[4]
    signal_args = params_tuple[5]
    backtest_args = params_tuple[6]

    # 从这些参数中取出索引为 idx 的子元素
    _data_args = data_args[idx, :]
    # 关键：从 UniTuple 中取出一个数组的切片
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


# 模拟 _parallel_calc
# 增加一个 out_array 参数用于写入结果
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
        ),
        nb.float64[:],  # 新增的输出数组参数
    ),
    cache=True,
)
def _parallel_calc_simulated(params_child, out_array_row):
    _data_args, indicator_args_child, _signal_args_child, _backtest_args_child = (
        params_child
    )

    # 简单的计算，例如：将一些扁平化参数的第一个元素求和，写入输出数组
    # 确保操作是安全的，例如检查数组长度
    sum_val = 0.0
    if _data_args.size > 0:
        sum_val += _data_args[0]
    if indicator_args_child[0].size > 0:  # UniTuple的第一个子数组
        sum_val += indicator_args_child[0][0]
    if _signal_args_child.size > 0:
        sum_val += _signal_args_child[0]

    # 写入结果到 out_array_row
    # 由于 out_array_row 是一个标量值，我们将其赋值给索引0
    # 在实际应用中，out_array_row 可能是更大的数组，这里简化为只写一个结果
    out_array_row[0] = sum_val  # 将计算结果写入输出数组的第一个元素


# --- 3. 核心测试函数 ---


def run_test_case(test_mode):
    print(f"\n--- Running Test Case: {test_mode} ---")

    _get_conf_count = nb.njit(nb.int64(nb.typeof(None)), cache=True)(lambda p: 100)
    _unpack_flatten = _unpack_flatten_simulated
    _unpack_flatten_idx = _unpack_flatten_idx_simulated  # 统一使用这个函数
    _parallel_calc = _parallel_calc_simulated

    params = None
    conf_count = _get_conf_count(params)
    res = _unpack_flatten(params)  # res 是一个包含 7 个元素的元组

    # 创建一个用于存储计算结果的输出数组
    # 每个 prange 迭代会计算一个结果，所以数组大小应为 conf_count
    # 这里我们简化为每个 prange 迭代只计算一个值，所以 output_results 应该是一维数组
    output_results = np.zeros(conf_count, dtype=np.float64)

    # Numba JIT 编译的内部循环函数
    # 保持 _cpu_parallel_calc_jit_inner 接收 FlattenedParamsType
    @nb.njit(
        nb.void(FlattenedParamsType, nb.float64[:]), parallel=True
    )  # 增加 output_results 参数
    def _cpu_parallel_calc_jit_inner_fixed(
        res_inner_complex_tuple, output_results_inner
    ):
        # --- 在 prange 外部，jit 内部进行解包 ---
        _data_args_unpacked = res_inner_complex_tuple[0]
        _indicator_params_unpacked = res_inner_complex_tuple[
            1
        ]  # 这是 UniTupleOfArrays4x2D
        _indicator_params2_unpacked = res_inner_complex_tuple[2]
        _indicator_result_unpacked = res_inner_complex_tuple[3]
        _indicator_result2_unpacked = res_inner_complex_tuple[4]
        _signal_args_unpacked = res_inner_complex_tuple[5]
        _backtest_args_unpacked = res_inner_complex_tuple[6]

        # 如果这里解包失败，会在编译 _cpu_parallel_calc_jit_inner_fixed 时报错

        # 初始化，避免 UnboundLocalError
        (
            _data_args_child,
            indicator_params_child,
            indicator_params2_child,
            indicator_result_child,
            indicator_result2_child,
            signal_args_child,
            backtest_args_child,
        ) = None, None, None, None, None, None, None

        for idx in nb.prange(conf_count):
            # --- 在 prange 内部，将已解包的变量重新封装，并传递给 _unpack_flatten_idx ---
            args_for_unpack_idx = None  # 初始化

            if test_mode == "JIT-Internal Direct Tuple Pass":
                args_for_unpack_idx = (
                    _data_args_unpacked,
                    _indicator_params_unpacked,
                    _indicator_params2_unpacked,
                    _indicator_result_unpacked,
                    _indicator_result2_unpacked,
                    _signal_args_unpacked,
                    _backtest_args_unpacked,
                )

            elif test_mode == "JIT-Internal Indexed Tuple Construction":
                # 方式 B: 通过索引从已解包的变量中构建元组
                # 注意：这里我们**故意保留**这个可能失败的模式来展示问题
                args_for_unpack_idx = (
                    res_inner_complex_tuple[0],
                    res_inner_complex_tuple[1],  # 这里就是那个 UniTuple
                    res_inner_complex_tuple[2],
                    res_inner_complex_tuple[3],
                    res_inner_complex_tuple[4],
                    res_inner_complex_tuple[5],
                    res_inner_complex_tuple[6],
                )

            elif test_mode == "JIT-Internal Implied Tuple From Unpack":
                # 方式 C: 在 prange 内部，对已解包的变量进行 Python 式的解包再打包
                # 注意：这里我们**故意保留**这个可能失败的模式来展示问题
                temp_da, temp_ip, temp_ip2, temp_ir, temp_ir2, temp_sa, temp_ba = (
                    res_inner_complex_tuple
                )
                args_for_unpack_idx = (
                    temp_da,
                    temp_ip,
                    temp_ip2,
                    temp_ir,
                    temp_ir2,
                    temp_sa,
                    temp_ba,
                )

            else:
                raise ValueError("Invalid test mode")

            # 调用 _unpack_flatten_idx，它只接收打包元组和索引
            (
                _data_args_child,
                indicator_params_child,
                indicator_params2_child,
                indicator_result_child,
                indicator_result2_child,
                signal_args_child,
                backtest_args_child,
            ) = _unpack_flatten_idx(args_for_unpack_idx, idx)

            # 后续的参数重组和调用保持不变
            indicator_args_child = (
                indicator_params_child,
                indicator_params2_child,
                indicator_result_child,
                indicator_result2_child,
            )
            _params_child = (
                _data_args_child,
                indicator_args_child,
                signal_args_child,
                backtest_args_child,
            )

            # 调用 _parallel_calc，并传入当前循环迭代的输出位置
            _parallel_calc(
                _params_child, output_results_inner[idx : idx + 1]
            )  # 传入一个切片，使其成为1D数组

    try:
        # 调用 Numba 编译后的内部函数，传入原始的复杂元组 res 和 output_results 数组
        _cpu_parallel_calc_jit_inner_fixed(res, output_results)
        print(f"Test Case '{test_mode}' SUCCEEDED.")

        # 验证计算结果
        expected_value = (
            1.0 + 1.0 + 1.0
        )  # 因为所有数组都用np.ones初始化，第一个元素都是1.0
        # 确保所有结果都符合预期
        if np.allclose(output_results, expected_value):
            print(
                f"Calculation for '{test_mode}' VERIFIED: All results are {expected_value}"
            )
        else:
            print(f"Calculation for '{test_mode}' FAILED: Results are not as expected.")
            print(f"Expected: {expected_value}, Got: {output_results}")

    except Exception as e:
        print(f"Test Case '{test_mode}' FAILED with error: {type(e).__name__}: {e}")


# --- 运行测试案例 ---
run_test_case("JIT-Internal Direct Tuple Pass")  # 正常
# run_test_case("JIT-Internal Indexed Tuple Construction")  # 预计会报错
# run_test_case("JIT-Internal Implied Tuple From Unpack") # 预计会报错
