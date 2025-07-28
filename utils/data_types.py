import numba as nb
import numpy as np


def get_numba_data_types(enable64: bool = True):
    np_int_type = np.int64 if enable64 else np.int32
    np_float_type = np.float64 if enable64 else np.float32
    nb_int_type = nb.int64 if enable64 else nb.int32
    nb_float_type = nb.float64 if enable64 else nb.float32
    dtype_dict = {
        "np": {
            "int": np_int_type,
            "float": np_float_type
        },
        "nb": {
            "int": nb_int_type,
            "float": nb_float_type
        }
    }
    return dtype_dict


default_types = get_numba_data_types(enable64=True)


def get_params_signature(nb_int_type, nb_float_type):
    return nb.types.Tuple((  # params
        nb.types.Tuple((  # data_args
            nb_float_type[:, :],  # tohlcv
            nb_float_type[:, :],  # tohlcv2
            nb_float_type[:, :],  # tohlcv_smooth
            nb_float_type[:, :],  # tohlcv_smooth2
        )),
        nb.types.Tuple((  # indicator_args
            # indicator_params
            nb.types.Tuple((
                nb_float_type[:, :],  # sma_params
                nb_float_type[:, :],  # sma2_params
                nb_float_type[:, :],  # bbands_params
            )),
            # indicator_params2
            nb.types.Tuple((
                nb_float_type[:, :],  # sma_params2
                nb_float_type[:, :],  # sma2_params2
                nb_float_type[:, :],  # bbands_params2
            )),
            nb_float_type[:],  # indicator_enabled
            nb_float_type[:],  # indicator_enabled2
            # indicator_result
            nb.types.Tuple((
                nb_float_type[:, :, :],  # sma_result
                nb_float_type[:, :, :],  # sma2_result
                nb_float_type[:, :, :],  # bbands_result
            )),
            # indicator_result2
            nb.types.Tuple((
                nb_float_type[:, :, :],  # sma_result2
                nb_float_type[:, :, :],  # sma2_result2
                nb_float_type[:, :, :],  # bbands_result2
            )),
        )),
        nb.types.Tuple((  # signal_args
            nb_float_type[:],  # signal_params
            nb_float_type[:, :, :],  # signal_result
        )),
        nb.types.Tuple((  # backtest_args
            nb_float_type[:, :],  # backtest_params
            nb_float_type[:, :, :],  # backtest_result
            nb_float_type[:, :, :],  # temp_arrays
        ))))


def get_params_child_signature(nb_int_type, nb_float_type):

    return nb.types.Tuple((  # params
        nb.types.Tuple((  # data_args
            nb_float_type[:, :],  # tohlcv
            nb_float_type[:, :],  # tohlcv2
            nb_float_type[:, :],  # tohlcv_smooth
            nb_float_type[:, :],  # tohlcv_smooth2
        )),
        nb.types.Tuple((  # indicator_args
            # indicator_params
            nb.types.Tuple((
                nb_float_type[:],  # sma_params_child
                nb_float_type[:],  # sma2_params_child
                nb_float_type[:],  # bbands_params_child
            )),
            # indicator_params2
            nb.types.Tuple((
                nb_float_type[:],  # sma_params2_child
                nb_float_type[:],  # sma2_params2_child
                nb_float_type[:],  # bbands_params2_child
            )),
            nb_float_type[:],  # indicator_enabled
            nb_float_type[:],  # indicator_enabled2
            # indicator_result
            nb.types.Tuple((
                nb_float_type[:, :],  # sma_result_child
                nb_float_type[:, :],  # sma2_result_child
                nb_float_type[:, :],  # bbands_result_child
            )),
            # indicator_result2
            nb.types.Tuple((
                nb_float_type[:, :],  # sma_result2_child
                nb_float_type[:, :],  # sma2_result2_child
                nb_float_type[:, :],  # bbands_result2_child
            )),
        )),
        nb.types.Tuple((  # signal_args
            nb_float_type[:],  # signal_params
            nb_float_type[:, :],  # signal_result_child
        )),
        nb.types.Tuple((  # backtest_args
            nb_float_type[:],  # backtest_params_child
            nb_float_type[:, :],  # backtest_result_child
            nb_float_type[:, :],  # temp_arrays_child
        ))))


def get_params_unpack_flatten(nb_int_type, nb_float_type):
    return nb.types.Tuple((  # indicator_args
        nb.types.Tuple((  # data_args
            nb_float_type[:, :],  # tohlcv
            nb_float_type[:, :],  # tohlcv2
            nb_float_type[:, :],  # tohlcv_smooth
            nb_float_type[:, :],  # tohlcv_smooth2
        )),
        # indicator_params
        nb.types.Tuple((
            nb_float_type[:, :],  # sma_params
            nb_float_type[:, :],  # sma2_params
            nb_float_type[:, :],  # bbands_params
        )),
        # indicator_params2
        nb.types.Tuple((
            nb_float_type[:, :],  # sma_params2
            nb_float_type[:, :],  # sma2_params2
            nb_float_type[:, :],  # bbands_params2
        )),
        # indicator_result
        nb.types.Tuple((
            nb_float_type[:, :, :],  # sma_result
            nb_float_type[:, :, :],  # sma2_result
            nb_float_type[:, :, :],  # bbands_result
        )),
        # indicator_result2
        nb.types.Tuple((
            nb_float_type[:, :, :],  # sma_result2
            nb_float_type[:, :, :],  # sma2_result2
            nb_float_type[:, :, :],  # bbands_result2
        )),
        nb.types.Tuple((  # signal_args
            nb_float_type[:, :],  # signal_params
            nb_float_type[:, :],  # signal_params2
            nb_float_type[:, :, :],  # signal_result
            nb_float_type[:, :, :],  # signal_result2
        )),
        nb.types.Tuple((  # backtest_args
            nb_float_type[:, :],  # backtest_params
            nb_float_type[:, :, :],  # backtest_result
            nb_float_type[:, :, :],  # temp_arrays
        ))))


def get_params_unpack_flatten_child(nb_int_type, nb_float_type):
    return nb.types.Tuple((
        nb.types.Tuple((  # data_args
            nb_float_type[:, :],  # tohlcv
            nb_float_type[:, :],  # tohlcv2
            nb_float_type[:, :],  # tohlcv_smooth
            nb_float_type[:, :],  # tohlcv_smooth2
        )),
        # indicator_params_child (切片后 1D)
        nb.types.Tuple((
            nb_float_type[:],  # sma_params_child
            nb_float_type[:],  # sma2_params_child
            nb_float_type[:],  # bbands_params_child
        )),
        # indicator_params2_child (切片后 1D)
        nb.types.Tuple((
            nb_float_type[:],  # sma_params2_child
            nb_float_type[:],  # sma2_params2_child
            nb_float_type[:],  # bbands_params2_child
        )),
        # indicator_result_child (切片后 2D)
        nb.types.Tuple((
            nb_float_type[:, :],  # sma_result_child
            nb_float_type[:, :],  # sma2_result_child
            nb_float_type[:, :],  # bbands_result_child
        )),
        # indicator_result2_child (切片后 2D)
        nb.types.Tuple((
            nb_float_type[:, :],  # sma_result2_child
            nb_float_type[:, :],  # sma2_result2_child
            nb_float_type[:, :],  # bbands_result2_child
        )),
        # signal_args_child (切片后)
        nb.types.Tuple((
            nb_float_type[:],  # signal_params_child
            nb_float_type[:],  # signal_params2_child
            nb_float_type[:, :],  # signal_result_child
            nb_float_type[:, :],  # signal_result2_child
        )),
        # backtest_args_child (切片后)
        nb.types.Tuple((
            nb_float_type[:],  # backtest_params_child
            nb_float_type[:, :],  # backtest_result_child
            nb_float_type[:, :],  # temp_arrays_child
        ))))
