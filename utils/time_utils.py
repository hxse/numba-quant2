import time
from functools import wraps


def time_wrapper(func):
    """
    一个装饰器，用于测量函数的执行时间。
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        prefix = kwargs["mode"] + " " if "mode" in kwargs else ""
        print(f"{prefix}{func.__name__} 运行时间: {run_time:.6f} 秒")
        return result

    return wrapper
