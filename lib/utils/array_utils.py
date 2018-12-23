import numpy as np


def wrap(value):
    if value is None:
        return []
    elif type(value) is list or type(value) is np.ndarray:
        return value
    else:
        return [value]

