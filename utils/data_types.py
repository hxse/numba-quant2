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


def get_signature(nb_int_type, nb_float_type):
    return nb.void(
        nb.types.Tuple((
            nb.types.Tuple((  # micro_data
                nb_float_type[:, :],  # micro_tohlcv
                nb_float_type[:, :],  # micro_tohlcv_smooth
            )),
            nb.types.Tuple((  # micro_input
                nb_float_type[:, :],  # micro_indicator_params
                nb_float_type[:, :],  # micro_signal_params
                nb_float_type[:, :, :],  # micro_indicator_result
                nb_float_type[:, :, :],  # micro_signal_result
            )),
            nb.types.Tuple((  # macro_data
                nb_float_type[:, :],  # macro_tohlcv
                nb_float_type[:, :],  # macro_tohlcv_smooth
            )),
            nb.types.Tuple((  # macro_input
                nb_float_type[:, :],  # macro_indicator_params
                nb_float_type[:, :],  # macro_signal_params
                nb_float_type[:, :, :],  # macro_indicator_result
                nb_float_type[:, :, :],  # macro_signal_result
            )),
            nb.types.Tuple((  # backtest_input
                nb_float_type[:, :],  # backtest_params
                nb_float_type[:, :, :],  # backtest_result
                nb_float_type[:, :, :],  # temp_arrays
            )))))


def get_signature_child(nb_int_type, nb_float_type):

    return nb.void(
        nb.types.Tuple((
            nb.types.Tuple((  # micro_data
                nb_float_type[:, :],  # micro_tohlcv
                nb_float_type[:, :],  # micro_tohlcv_smooth
            )),
            nb.types.Tuple((  # micro_input_child
                nb_float_type[:],  # micro_indicator_params_child
                nb_float_type[:],  # micro_signal_params_child
                nb_float_type[:, :],  # micro_indicator_result_child
                nb_float_type[:, :],  # micro_signal_result_child
            )),
            nb.types.Tuple((  # macro_data
                nb_float_type[:, :],  # macro_tohlcv
                nb_float_type[:, :],  # macro_tohlcv_smooth
            )),
            nb.types.Tuple((  # macro_input_child
                nb_float_type[:],  # macro_indicator_params_child
                nb_float_type[:],  # macro_signal_params_child
                nb_float_type[:, :],  # macro_indicator_result_child
                nb_float_type[:, :],  # macro_signal_result_child
            )),
            nb.types.Tuple((  # backtest_input_child
                nb_float_type[:],  # backtest_params_child
                nb_float_type[:, :],  # backtest_result_child
                nb_float_type[:, :],  # temp_arrays_child
            )))))
