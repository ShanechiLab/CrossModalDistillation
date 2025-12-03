import logging
import matplotlib
import numpy as np
import torch.nn as nn
import torch


def get_activation_function(activation_str):
    """
    Returns activation function given the activation function's name

    Parameters:
    ----------------------
    activation_str: str, Activation function's name

    Returns:
    ----------------------
    activation_fn: Activation function from torch.nn
    """

    if activation_str.lower() == "elu":
        return nn.ELU()
    elif activation_str.lower() == "hardtanh":
        return nn.Hardtanh()
    elif activation_str.lower() == "leakyrelu":
        return nn.LeakyReLU()
    elif activation_str.lower() == "relu":
        return nn.ReLU()
    elif activation_str.lower() == "rrelu":
        return nn.RReLU()
    elif activation_str.lower() == "sigmoid":
        return nn.Sigmoid()
    elif activation_str.lower() == "mish":
        return nn.Mish()
    elif activation_str.lower() == "tanh":
        return nn.Tanh()
    elif activation_str.lower() == "tanhshrink":
        return nn.Tanhshrink()
    elif activation_str.lower() == "linear":
        return lambda x: x
    elif activation_str.lower() == "silu":
        return nn.SiLU()
    elif activation_str.lower() == "gelu":
        return nn.GELU()


def set_matplotlib_starter_nature():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    font = {"family": "sans-serif", "weight": "normal", "size": 15}

    matplotlib.rc("font", **font)
    return


def set_plot_settings_nature(fig, ax_list):
    for ax in ax_list:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.tick_params("both", length=8, width=1, which="major", color="black")
        ax.tick_params("both", length=4, width=0.5, which="minor", color="black")

    return


def get_closest_indices(array1, array2):
    array1 = np.array(array1, dtype=object)
    array2 = np.array(array2)

    # Find the index of the closest element in array2 for each element in array1
    closest_indices = [
        np.abs(array2 - value).argmin() if value is not None else None
        for value in array1
    ]
    return closest_indices


def int2str(int_arr):
    s = ""
    for i in int_arr.reshape(-1):
        s += chr(i)
    return s


LOG_BASE_FORMAT = "%(asctime)s  [%(name)s]  %(levelname)8s | %(message)s"


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = LOG_BASE_FORMAT

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(name="CrossModalDistillation", log_level="debug"):
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.DEBUG,
        "error": logging.DEBUG,
    }
    # from: https://stackoverflow.com/a/56689445/16228104
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # remove old handlers from logger (since logger is static object) so that in several calls, it doesn't overwrite to previous log files
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(log_level_map[log_level])

    # create formatter and add it to the handlers
    ch.setFormatter(CustomFormatter())

    # add the handlers to logger
    logger.addHandler(ch)

    # disables multiple logging
    logger.propagate = False

    return logger


def to_cuda(obj, device=None, non_blocking=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)

    elif isinstance(obj, dict):
        return {k: to_cuda(v, device, non_blocking) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        out = [to_cuda(x, device, non_blocking) for x in obj]
        return type(obj)(out)

    else:
        return obj


def to_dtype(obj, dtype="float32"):
    dtype_obj = getattr(torch, dtype)

    if torch.is_tensor(obj):
        return obj.to(dtype_obj)

    elif isinstance(obj, dict):
        return {k: to_dtype(v, dtype) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        out = [to_dtype(x, dtype) for x in obj]
        return type(obj)(out)

    else:
        return obj
