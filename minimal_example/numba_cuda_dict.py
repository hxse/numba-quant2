import numpy as np
from numba import njit, cuda
from numba.core import types
from numba.typed import Dict

print("--- CPU (njit) Mode Example ---")

# The Dict.empty() constructs a typed dictionary.
# The key and value typed must be explicitly declared.
d_cpu = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64[:],
)

# The typed-dict can be used from the interpreter.
d_cpu["posx"] = np.asarray([1, 0.5, 2], dtype="f8")
d_cpu["posy"] = np.asarray([1.5, 3.5, 2], dtype="f8")
d_cpu["velx"] = np.asarray([0.5, 0, 0.7], dtype="f8")
d_cpu["vely"] = np.asarray([0.2, -0.2, 0.1], dtype="f8")


# Here's a function that expects a typed-dict as the argument
@njit
def move_cpu(d):
    # inplace operations on the arrays
    d["posx"] += d["velx"]
    d["posy"] += d["vely"]


print("CPU Initial posx: ", d_cpu["posx"])
print("CPU Initial posy: ", d_cpu["posy"])

# Call move_cpu(d_cpu) to inplace update the arrays in the typed-dict.
move_cpu(d_cpu)

print("CPU Updated posx: ", d_cpu["posx"])
print("CPU Updated posy: ", d_cpu["posy"])

print("\n" + "=" * 50 + "\n")  # 分隔线

print("--- GPU (cuda.jit) Mode Example (Expected to FAIL) ---")

if not cuda.is_available():
    print("CUDA device not found. Skipping GPU example.")
else:
    print("CUDA device available.")

    # 1. 准备数据：创建 numba.typed.Dict
    #    请注意：这个字典本身仍在CPU内存中。
    d_gpu_typed_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64[:],
    )
    d_gpu_typed_dict["posx"] = np.asarray([1, 0.5, 2], dtype="f8")
    d_gpu_typed_dict["posy"] = np.asarray([1.5, 3.5, 2], dtype="f8")
    d_gpu_typed_dict["velx"] = np.asarray([0.5, 0, 0.7], dtype="f8")
    d_gpu_typed_dict["vely"] = np.asarray([0.2, -0.2, 0.1], dtype="f8")

    print("GPU Initial (from Dict) posx: ", d_gpu_typed_dict["posx"])
    print("GPU Initial (from Dict) posy: ", d_gpu_typed_dict["posy"])

    # 2. 定义 CUDA 核函数 - **直接尝试传入 numba.typed.Dict**
    #    这会触发 Numba 的类型推断错误或编译错误，因为它无法将Dict映射到GPU类型。
    @cuda.jit
    def move_gpu_kernel_with_dict(d_device_dict):  # 尝试接收一个字典
        idx = cuda.grid(1)

        if idx < d_device_dict["posx"].shape[0]:  # 尝试访问字典元素
            d_device_dict["posx"][idx] += d_device_dict["velx"][idx]
            d_device_dict["posy"][idx] += d_device_dict["vely"][idx]

    # 3. 在 Python 端调用 CUDA 核函数
    #    尝试将字典（或其设备版本）传入
    try:
        # Numba 不支持将 numba.typed.Dict 完整地复制到设备内存。
        # cuda.to_device(d_gpu_typed_dict) 会失败。
        # 所以，我们甚至无法达到调用核函数的步骤，因为数据传输就卡住了。
        # 即使 d_gpu_typed_dict 本身是在CPU，Numba 也会在编译 move_gpu_kernel_with_dict 时发现问题。

        # 尝试直接调用，Numba 会在 JIT 编译 move_gpu_kernel_with_dict 时抛出错误
        # 因为它无法为 'd_device_dict' 参数推断出 CUDA 兼容的类型。
        print("\nAttempting to compile and run CUDA kernel with numba.typed.Dict...")

        # 为了触发错误，我们至少需要尝试编译。
        # 定义一个简单的调用函数来触发核函数编译
        def run_move_gpu_with_dict(input_dict):
            # 实际上，`cuda.to_device(input_dict)` 这一步很可能就会失败，
            # 或者在核函数编译时 Numba 就会报错。
            # 我们这里直接尝试传递 Python 字典对象给核函数，
            # 让 Numba 在编译时尝试类型推断。

            # 即使不显式 `to_device`，Numba 也会尝试在编译时验证参数类型
            # 而 numba.typed.Dict 无法直接映射到 GPU 上。
            data_size = input_dict["posx"].shape[0]
            threads_per_block = 256
            blocks_per_grid = (data_size + (threads_per_block - 1)) // threads_per_block

            move_gpu_kernel_with_dict[blocks_per_grid, threads_per_block](input_dict)

        run_move_gpu_with_dict(d_gpu_typed_dict)

        # 如果代码能运行到这里，说明我之前的理解有误，但根据Numba文档，这几乎不可能
        print("CUDA kernel with numba.typed.Dict ran successfully (unexpected).")

        # 由于可能根本无法运行，以下更新和验证的代码可能无法到达
        print("GPU Updated (from Dict) posx: ", d_gpu_typed_dict["posx"])
        print("GPU Updated (from Dict) posy: ", d_gpu_typed_dict["posy"])

    except Exception as e:
        print("\n!!! EXPECTED ERROR OCCURRED !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print(
            "\nThis confirms that numba.typed.Dict is NOT directly supported in cuda.jit kernels."
        )
