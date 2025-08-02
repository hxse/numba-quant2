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
  * 这个`numba-quant2`比`num-quant`封装的更好, 但是依然难写
  * 参考`minimal_example\numba_cache_example.py`, numba闭包时,有个令人讨厌的硬盘缓存问题,算了,不折腾了
