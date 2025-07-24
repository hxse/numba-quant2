# ncu
  * `ncu "C:\Users\qmlib\scoop\apps\uv\current\uv.exe" run .\src\example\example.py`
# run
  * `uv run .\src\main.py`
# todo
  * 这个`numba-quant2`比`num-quant`封装的更好, 但是依然难写
  * 比如,需要构建指标的动态加载和信号生成的依赖指标
  * 添加指标时,需要提取参数和结果数组,需要硬编码函数,样板代码多,容易出错
  * 添加信号生成时,需要提取结果数组,需要硬编码函数,样板代码多,容易出错
  * 先放下了,写Jax或PyTorch去了
