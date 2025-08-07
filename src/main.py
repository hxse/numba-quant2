import sys
from pathlib import Path

root_path = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
    None,
)
if root_path:
    sys.path.insert(0, str(root_path))

from utils.export_file import export_csv
from utils.json_tool import load_numba_config
import time
import typer


def main(
    mode: str = "njit",
    cache: bool = True,
    enable64: bool = True,
    pre_run: bool = True,
    core_time: bool = True,
    task_time: bool = True,
    total_time: bool = False,
    max_registers: int = 24,
):
    """
    pre_run 控制是否执行第一次迭代 (预运行)
    total_time 包含jit,njit,cuda的完整运行时间,不包括csv文件导入时间
    task_time 包含数据预生成和内核运行的时间
    core_time 内核运行的时间
    """
    # 加载numba_config临时配置文件
    nb_params = load_numba_config(
        mode=mode, cache=cache, enable64=enable64, max_registers=max_registers
    )

    # 确保加载完numba_config后再import numba
    start_time = time.time()
    import numba as nb
    from src.interface import entry_func, entry_func_wrapper
    from utils.config_utils import get_dtype_dict, perpare_data, get_params
    from src.indicators.sma import (
        sma_id,
        sma2_id,
        sma_name,
        sma2_name,
        sma_spec,
        sma2_spec,
    )
    from src.indicators.bbands import bbands_id, bbands_name, bbands_spec
    from src.indicators.atr import atr_id, atr_name, atr_spec
    from src.indicators.psar import psar_id, psar_name, psar_spec
    from src.signal.simple_template import simple_id, simple_name
    from src.backtest.calculate_backtest import backtest_result_name
    from src.calculate_signals import signal_result_name

    end_time = time.time()
    print(f"numba模块导入冷启动时间: {end_time - start_time:.4f} 秒")

    print("#### CUDA 可用性检测 ####")
    if nb.cuda.is_available():
        print("CUDA 可用")
    else:
        print("CUDA 不可用")
        exit()

    for i in range(2):
        # 如果是第一次迭代 (i=0) 并且 pre_run 为 False，则跳过当前循环
        if i == 0 and not pre_run:
            continue

        path = "database/live/BTC_USDT/15m/BTC_USDT_15m_20230228 160000.csv"
        path2 = "database/live/BTC_USDT/4h/BTC_USDT_4h_20230228 160000.csv"

        data_size = 40 * 1000
        data_size = 10 if i == 0 else data_size

        dtype_dict = get_dtype_dict(enable64=True)

        df_data, np_data = perpare_data(
            path, data_size=data_size, dtype_dict=dtype_dict
        )
        df_data2, np_data2 = perpare_data(
            path2, data_size=data_size, dtype_dict=dtype_dict
        )
        num = 1
        params = get_params(
            num=num,
            indicator_update={
                sma_name: [[14] for i in range(num)],
                sma2_name: [[50] for i in range(num)],
                bbands_name: [[20, 2.0] for i in range(num)],
                atr_name: [[14] for i in range(num)],
                psar_name: [[0.02, 0.02, 0.2] for i in range(num)],
            },
            indicator_enabled={
                # sma_name: True,
                # sma2_name: True,
                # bbands_name: True,
                # atr_name: True,
                # psar_name: True,
            },
            signal_name="simple",
            backtest_params={
                "pct_sl_enable": False,
                "pct_tp_enable": False,
                "pct_tsl_enable": False,
                "pct_sl": 0.01,
                "pct_tp": 0.01,
                "pct_tsl": 0.01,
                "atr_sl_enable": False,
                "atr_tp_enable": False,
                "atr_tsl_enable": False,
                "atr_preiod": 14,
                "atr_sl_multiplier": 2.0,
                "atr_tp_multiplier": 2.0,
                "atr_tsl_multiplier": 2.0,
                "psar_enable": False,
                "psar_af0": 0.02,
                "psar_af_step": 0.02,
                "psar_max_af": 0.2,
            },
            indicator_update2={
                sma_name: [[100] for i in range(num)],
                sma2_name: [[200] for i in range(num)],
                bbands_name: [[20, 2.0] for i in range(num)],
                atr_name: [[14] for i in range(num)],
                psar_name: [[0.02, 0.02, 0.2] for i in range(num)],
            },
            indicator_enabled2={
                # sma_name: True,
                # sma2_name: True,
                # bbands_name: True,
                # atr_name: True,
                # psar_name: True,
            },
            dtype_dict=dtype_dict,
        )

        print(
            f"\nstart task idx: {i}",
            "并发数量:",
            len(params["backtest_params"]),
            "数据数量:",
            len(np_data),
        )
        print("numba_params:", nb_params)

        start_time = time.time()

        mode = nb_params.get("mode", "njit")

        _func = entry_func_wrapper if task_time else entry_func
        (
            tohlcv,
            tohlcv2,
            mapping_data,
            indicator_result,
            indicator_result2,
            signal_result,
            backtest_result,
            int_temp_array,
            float_temp_array,
            bool_temp_array,
        ) = _func(
            mode,
            np_data,
            params["indicator_params"],
            params["indicator_enabled"],
            params["signal_params"],
            params["backtest_params"],
            tohlcv2=np_data2,
            indicator_params2=params["indicator_params2"],
            indicator_enabled2=params["indicator_enabled2"],
            mapping_data=params["mapping_data"],
            cache=cache,
            dtype_dict=dtype_dict,
            core_time=core_time,
        )

        if i != 0:
            print(f"{mode} out_arrays length:", len(backtest_result))
            # print(f"{mode} indicator_result:", indicator_result)
            # print(f"{mode} indicator_result2:", indicator_result2)
            # print(f"{mode} signal_result:", signal_result)

            export_csv(
                simple_name,
                [["time", "open", "high", "low", "close", "volume"], tohlcv],
                [["time", "open", "high", "low", "close", "volume"], tohlcv2],
                [params["indicator_col_name"], indicator_result],
                [params["indicator_col_name2"], indicator_result2],
                [signal_result_name, signal_result],
                [backtest_result_name, backtest_result],
                params["indicator_enabled"],
                params["indicator_enabled2"],
                write_csv=True,
            )

        if total_time:
            print(f"Task {i} total_time: {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_show_locals=False)
    app.command()(main)
    app()
