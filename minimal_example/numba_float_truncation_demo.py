import numba as nb
import numpy as np
import os
import shutil  # 用于删除文件夹


# --- 核心 SMA 计算逻辑 (JIT 函数) ---
# 签名严格要求 sma_period 为 int64 类型
@nb.jit(nb.void(nb.float64[:], nb.int64, nb.float64[:]), nopython=True)
def calculate_sma_core(close, sma_period, sma_result):
    """
    核心 SMA 计算函数。
    它要求 sma_period 必须是 int64 类型。
    """
    # 填充前 sma_period - 1 个元素为 NaN
    for i in range(min(sma_period - 1, len(close))):
        sma_result[i] = np.nan

    # 从 sma_period - 1 索引开始计算 SMA
    for i in range(len(close) - sma_period + 1):
        sum_val = 0.0
        for j in range(sma_period):
            sum_val += close[i + j]
        sma_result[i + sma_period - 1] = sum_val / sma_period


# --- 测试函数 ---
def test_numba_float_to_int_conversion(sma_period_input):
    """
    测试 Numba 在 nopython 模式下，当函数签名要求 int64 但传入 float64 时的行为。
    """
    close = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64
    )
    sma_result = np.zeros_like(close)

    print(
        f"\n--- 测试 SMA 周期: {sma_period_input} (类型: {type(sma_period_input).__name__}) ---"
    )
    print(f"输入收盘价 (close): {close}")

    try:
        # 尝试直接将 float 类型的 sma_period_input 传递给期望 int64 的 JIT 函数
        calculate_sma_core(close, sma_period_input, sma_result)
        print(f"计算结果 (sma_result): {sma_result}")

        # 如果成功，验证 sma_period_input 是如何被“隐式”处理的
        if isinstance(sma_period_input, float):
            expected_period_int = int(sma_period_input)  # 预期 Numba 内部会做的截断
            print(
                f"🤔 Numba 内部可能将 {sma_period_input} 转换成了整数: {expected_period_int}"
            )

            # 重新计算预期结果，以验证是否是截断后的值
            expected_sma_result = np.full_like(close, np.nan)
            if len(close) >= expected_period_int:
                for i in range(len(close) - expected_period_int + 1):
                    expected_sma_result[i + expected_period_int - 1] = (
                        np.sum(close[i : i + expected_period_int]) / expected_period_int
                    )

            # 对于 NaN 的比较需要特殊处理
            are_equal = np.allclose(sma_result, expected_sma_result, equal_nan=True)
            print(
                f"计算结果是否符合基于截断整数 ({expected_period_int}) 的预期？ {are_equal}"
            )

        print("✅ 成功！Numba 接受了浮点数输入并正确执行。")
        print(
            "这表明 Numba 能够安全地将此浮点数转换为整数（无精度损失，或进行了可接受的截断）。"
        )

    except nb.core.errors.TypingError as e:
        print("\n❌ 捕获到 TypingError (预期行为) ---")
        print("错误信息：")
        print(e)
        print(
            "\n这证实了 Numba 在 nopython 模式下，不会将具有小数部分的 float 隐式转换为 int，因为它会导致精度损失。"
        )
    except Exception as e:
        print(f"\n--- 捕获到其他错误 ---")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")


if __name__ == "__main__":
    print("--- 尝试清除 Numba 缓存... ---")
    # 尝试删除 Numba 缓存目录，这通常是解决这种“意外”行为的最佳方法
    # 注意：根据你的环境和 Numba 版本，缓存目录的位置可能有所不同。
    # 确保这个路径正确指向你的 Numba 编译缓存。
    numba_cache_base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "__pycache__", "numba_cache"
    )

    # 查找所有以 .pyc 开头并包含 numba_cache 的子目录
    # 这种方式更健壮，因为 .numba 缓存可能在多个位置
    # 遍历当前脚本所在目录的所有子目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    deleted_any_cache = False
    for root, dirs, files in os.walk(current_dir):
        for d in dirs:
            if d == "__pycache__":
                pycache_path = os.path.join(root, d)
                for item in os.listdir(pycache_path):
                    item_path = os.path.join(pycache_path, item)
                    if os.path.isdir(item_path) and "numba_cache" in item_path:
                        try:
                            shutil.rmtree(item_path)
                            print(f"已清除 Numba 缓存目录: {item_path}")
                            deleted_any_cache = True
                        except Exception as e:
                            print(f"无法清除 {item_path}: {e}")

    if not deleted_any_cache:
        print("未找到 Numba 缓存目录，无需清除。")

    test_numba_float_to_int_conversion(3.0)  # 测试整数值浮点数
    test_numba_float_to_int_conversion(3.7)  # 测试带小数的浮点数 (这次是 3.7)
