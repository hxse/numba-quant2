import numba as nb
import numpy as np


def get_numba_data_types(enable64: bool = True):
    np_int_type = np.int64 if enable64 else np.int32
    np_float_type = np.float64 if enable64 else np.float32
    np_bool_type = np.bool_
    nb_int_type = nb.int64 if enable64 else nb.int32
    nb_float_type = nb.float64 if enable64 else nb.float32
    nb_bool_type = nb.boolean
    dtype_dict = {
        "np": {"int": np_int_type, "float": np_float_type, "bool": np_bool_type},
        "nb": {"int": nb_int_type, "float": nb_float_type, "bool": nb_bool_type},
    }
    return dtype_dict


def get_indicator_params(nb_int_type, nb_float_type, nb_bool_type):
    return nb.types.Tuple(
        (
            nb_float_type[:, :],  # sma_params
            nb_float_type[:, :],  # sma2_params
            nb_float_type[:, :],  # bbands_params
            nb_float_type[:, :],  # atr_params
            nb_float_type[:, :],  # psar_params
        )
    )


def get_indicator_params_child(nb_int_type, nb_float_type, nb_bool_type):
    return nb.types.Tuple(
        (
            nb_float_type[:],  # sma_params_child
            nb_float_type[:],  # sma2_params_child
            nb_float_type[:],  # bbands_params_child
            nb_float_type[:],  # atr_params_child
            nb_float_type[:],  # psar_params_child
        )
    )


def get_indicator_result(nb_int_type, nb_float_type, nb_bool_type):
    return nb.types.Tuple(
        (
            nb_float_type[:, :, :],  # sma_result
            nb_float_type[:, :, :],  # sma2_result
            nb_float_type[:, :, :],  # bbands_result
            nb_float_type[:, :, :],  # atr_result
            nb_float_type[:, :, :],  # psar_result
        )
    )


def get_indicator_result_child(nb_int_type, nb_float_type, nb_bool_type):
    return nb.types.Tuple(
        (
            nb_float_type[:, :],  # sma_result_child
            nb_float_type[:, :],  # sma2_result_child
            nb_float_type[:, :],  # bbands_result_child
            nb_float_type[:, :],  # atr_result_child
            nb_float_type[:, :],  # psar_result_child
        )
    )


def get_temp_result(nb_int_type, nb_float_type, nb_bool_type):
    return nb.types.Tuple(
        (
            nb_int_type[:, :, :],  # int_temp_array
            nb_int_type[:, :, :],  # int_temp_array2
            nb_float_type[:, :, :],  # float_temp_array
            nb_float_type[:, :, :],  # float_temp_array2
            nb_bool_type[:, :, :],  # bool_temp_array
            nb_bool_type[:, :, :],  # bool_temp_array2
        )
    )


def get_temp_result_child(nb_int_type, nb_float_type, nb_bool_type):
    return nb.types.Tuple(
        (
            nb_int_type[:, :],  # int_temp_array_child
            nb_int_type[:, :],  # int_temp_array2_child
            nb_float_type[:, :],  # float_temp_array_child
            nb_float_type[:, :],  # float_temp_array2_child
            nb_bool_type[:, :],  # bool_temp_array_child
            nb_bool_type[:, :],  # bool_temp_array2_child
        )
    )


def get_params_signature(nb_int_type, nb_float_type, nb_bool_type):
    return nb.types.Tuple(
        (  # params
            nb.types.Tuple(
                (  # data_args
                    nb_float_type[:, :],  # tohlcv
                    nb_float_type[:, :],  # tohlcv2
                    nb_float_type[:, :],  # tohlcv_smooth
                    nb_float_type[:, :],  # tohlcv_smooth2
                    nb_int_type[:],  # mapping_data
                )
            ),
            nb.types.Tuple(
                (  # indicator_args
                    # indicator_params
                    get_indicator_params(nb_int_type, nb_float_type, nb_bool_type),
                    # indicator_params2
                    get_indicator_params(nb_int_type, nb_float_type, nb_bool_type),
                    nb_bool_type[:],  # indicator_enabled
                    nb_bool_type[:],  # indicator_enabled2
                    # indicator_result
                    get_indicator_result(nb_int_type, nb_float_type, nb_bool_type),
                    # indicator_result2
                    get_indicator_result(nb_int_type, nb_float_type, nb_bool_type),
                )
            ),
            nb.types.Tuple(
                (  # signal_args
                    nb_int_type[:],  # signal_params
                    nb_bool_type[:, :, :],  # signal_result
                )
            ),
            nb.types.Tuple(
                (  # backtest_args
                    nb_float_type[:, :],  # backtest_params
                    nb_float_type[:, :, :],  # backtest_result
                )
            ),
            nb.types.Tuple(
                (  # temp_args
                    get_temp_result(nb_int_type, nb_float_type, nb_bool_type)
                )
            ),
        )
    )


def get_params_child_signature(nb_int_type, nb_float_type, nb_bool_type):
    return nb.types.Tuple(
        (  # params
            nb.types.Tuple(
                (  # data_args
                    nb_float_type[:, :],  # tohlcv
                    nb_float_type[:, :],  # tohlcv2
                    nb_float_type[:, :],  # tohlcv_smooth
                    nb_float_type[:, :],  # tohlcv_smooth2
                    nb_int_type[:],  # mapping_data
                )
            ),
            nb.types.Tuple(
                (  # indicator_args
                    # indicator_params
                    get_indicator_params_child(
                        nb_int_type, nb_float_type, nb_bool_type
                    ),
                    # indicator_params2
                    get_indicator_params_child(
                        nb_int_type, nb_float_type, nb_bool_type
                    ),
                    nb_bool_type[:],  # indicator_enabled
                    nb_bool_type[:],  # indicator_enabled2
                    # indicator_result
                    get_indicator_result_child(
                        nb_int_type, nb_float_type, nb_bool_type
                    ),
                    # indicator_result2
                    get_indicator_result_child(
                        nb_int_type, nb_float_type, nb_bool_type
                    ),
                )
            ),
            nb.types.Tuple(
                (  # signal_args
                    nb_int_type[:],  # signal_params
                    nb_bool_type[:, :],  # signal_result_child
                )
            ),
            nb.types.Tuple(
                (  # backtest_args
                    nb_float_type[:],  # backtest_params_child
                    nb_float_type[:, :],  # backtest_result_child
                )
            ),
            nb.types.Tuple(
                (  # temp_args
                    get_temp_result_child(nb_int_type, nb_float_type, nb_bool_type)
                )
            ),
        )
    )


def get_indicator_wrapper_signal(nb_int_type, nb_float_type, nb_bool_type):
    return (
        nb_float_type[:, :],  # tohlcv
        get_indicator_params_child(
            nb_int_type, nb_float_type, nb_bool_type
        ),  # indicator_params_child
        get_indicator_result_child(
            nb_int_type, nb_float_type, nb_bool_type
        ),  # indicator_result_child
        nb_float_type[:, :],  # float_temp_array_child
        nb_int_type,  # _id
    )
