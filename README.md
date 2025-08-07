# ncu
  * `ncu "C:\Users\qmlib\scoop\apps\uv\current\uv.exe" run .\src\example\example.py`
# run
  * `uv run .\src\main.py`
# 添加指标
  * `src\indicators`文件夹下创建指标文件
  * `utils\config_utils.py`文件下,修改`get_indicator_params`和`indicator_count`
  * `src\main.py`文件下,修改`get_params`传参
  * `utils\numba_unpack.py`文件下,修改`initialize_outputs`和`unpack_params_child`
  * `utils\data_types.py`文件下,修改`indicator_params`和`indicator_result`和`indicator_params_child`和`indicator_result_child`
  * `src\calculate_indicators.py`文件下,修改`calc_indicators`和`loop_indicators`
# todo
  * `utils\numba_unpack.py`文件下的initialize_outputs函数
    * 需要弄一个全局缓存,复用结果数组的内存,在numba内核函数中初始化
    * 如果数组形状不同,就新增缓存,用一个最大次数参数去,控制最大缓存多少个结果数组
