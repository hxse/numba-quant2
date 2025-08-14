global_outputs = []


def find_outputs(lookup_dict, global_outputs, find_key="outputs"):
    """
    根据给定的字典查找并返回 global_outputs 中匹配项的指定键的值，
    不创建任何字典副本。

    参数:
    - lookup_dict (dict): 用于查找的字典，不包含 'find_key' 指定的键。
    - global_outputs (list): 全局列表，其中每个元素都是一个字典。
    - find_key (str, optional): 默认为 "outputs"，指定要返回的键名。

    返回:
    - 匹配项的 find_key 对应的值，如果未找到则返回 None。
    """
    # 查找字典的键集合
    lookup_keys = set(lookup_dict.keys())

    # 遍历 global_outputs 中的每一个字典
    for item in global_outputs:
        # 获取当前项的键集合，并移除 find_key 以进行比较
        item_keys = set(item.keys())
        item_keys.discard(find_key)

        # 1. 首先检查键的数量和内容是否完全匹配
        if lookup_keys != item_keys:
            continue

        # 2. 如果键匹配，则逐个检查对应的值是否也匹配
        match = True
        for key in lookup_keys:
            if item.get(key) != lookup_dict.get(key):
                match = False
                break

        # 3. 如果所有键和值都匹配，则找到了，返回其 find_key 对应的值
        if match:
            return item.get(find_key)

    # 如果遍历完所有项都未找到匹配，则返回 None
    return None


def get_outputs_from_global(lookup_dict: dict):
    outputs = find_outputs(lookup_dict, global_outputs, "outputs")
    return outputs


def set_outputs_from_global(lookup_dict, outputs, max_size=1):
    """
    将新的结果字典添加到全局缓存中，并根据 max_size 进行缓存淘汰。

    参数:
    - lookup_dict (dict): 用于查找的字典。
    - outputs: 要缓存的结果对象。
    - max_size (int, optional): 缓存的最大容量。默认为 MAX_CACHE_SIZE。
    """
    # 将新的字典添加到全局列表中
    new_item = {**lookup_dict, "outputs": outputs}
    global_outputs.append(new_item)

    # 如果列表超过最大容量，则移除最旧（第一个）的元素
    if len(global_outputs) > max_size:
        global_outputs.pop(0)
