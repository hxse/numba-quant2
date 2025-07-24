from .sma import sma_id, sma_spec


def find_idx(s):
    if s == sma_spec["name"]:
        return sma_spec["id"]
    raise RuntimeError(f"没找到指标id {s}")


def get_indicators_spec(enable_array=[], dependency_array=[]):
    '''
    enable_array, 启用哪些指标(表示用户需要的指标), 可以接受指标id, 指标name, name会自动转换成id
    dependency_array, 启用哪些指标(表示信号生成依赖的指标), 可以接受指标id, 指标name, name会自动转换成id
    如果这两个数组都为空, 那么默认不加载任何指标
    '''
    for k, v in enumerate(enable_array):
        if type(v) == str:
            enable_array[k] = find_idx(v)

    for k, v in enumerate(dependency_array):
        if type(v) == str:
            dependency_array[k] = find_idx(v)


if __name__ == '__main__':
    get_indicators_spec(enable_array=["sma"])
