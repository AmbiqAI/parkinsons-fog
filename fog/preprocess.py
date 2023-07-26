import math
import scipy.signal as sps
import scipy.interpolate as spi
import tensorflow as tf
import numpy as np
import numpy.typing as npt

def sample_normalize(sample):
    mean = tf.math.reduce_mean(sample)
    std = tf.math.reduce_std(sample)
    sample = tf.math.divide_no_nan(sample-mean, std)
    return sample.numpy()

def resample_categorical(data: npt.NDArray, sample_rate: float, target_rate: float, axis: int = 0) -> npt.NDArray:
    """Resample categorical data using nearest neighbor.

    Args:
        data (npt.NDArray): Signal
        sample_rate (float): Signal sampling rate
        target_rate (float): Target sampling rate
        axis (int, optional): Axis to resample along. Defaults to 0.

    Returns:
        npt.NDArray: Resampled signal
    """
    if sample_rate == target_rate:
        return data
    ratio = target_rate / sample_rate
    actual_length = data.shape[axis]
    target_length = int(np.round(data.shape[axis] * ratio))
    interp_fn = spi.interp1d(np.arange(0, actual_length), data, kind='nearest', axis=axis)
    return interp_fn(np.arange(0, target_length)).astype(data.dtype)

def resample_signal(data: npt.NDArray, sample_rate: float, target_rate: float, axis: int = 0) -> npt.NDArray:
    """Resample signal using scipy FFT-based resample routine.

    Args:
        data (npt.NDArray): Signal
        sample_rate (float): Signal sampling rate
        target_rate (float): Target sampling rate
        axis (int, optional): Axis to resample along. Defaults to 0.

    Returns:
        npt.NDArray: Resampled signal
    """
    if sample_rate == target_rate:
        return data
    ratio = target_rate / sample_rate
    desired_length = int(np.round(data.shape[axis] * ratio))
    return sps.resample(data, desired_length, axis=axis)

def normalize_signal(data: npt.NDArray, eps: float = 1e-3, axis: int = 0) -> npt.NDArray:
    """Normalize signal about its mean and std.

    Args:
        data (npt.NDArray): Signal
        eps (float, optional): Epsilon added to st. dev. Defaults to 1e-3.
        axis (int, optional): Axis to normalize along. Defaults to 0.

    Returns:
        npt.NDArray: Normalized signal
    """
    mu = np.nanmean(data, axis=axis)
    std = np.nanstd(data, axis=axis) + eps
    return (data - mu) / std


# def get_blocks(
#     series,
#     columns,
#     block_size,
#     block_stride
# ):
#     series = series.copy()
#     series = series[columns]
#     series = series.values
#     series = series.astype(np.float32)

#     block_count = math.ceil(len(series) / block_size)

#     series = np.pad(series, pad_width=[[0, block_count*block_size-len(series)], [0, 0]])

#     block_begins = list(range(0, len(series), block_stride))
#     block_begins = [x for x in block_begins if x+block_size <= len(series)]

#     blocks = []
#     for begin in block_begins:
#         values = series[begin:begin+block_size]
#         blocks.append({'begin': begin, 'end': begin+block_size, 'values': values})
#     # END FOR
#     return blocks
