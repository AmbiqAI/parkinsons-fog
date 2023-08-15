import math
import functools

import scipy.signal as sps
import scipy.interpolate as spi
import numpy as np
import numpy.typing as npt


@functools.cache
def get_butter_sos(
    lowcut: float | None = None,
    highcut: float | None = None,
    sample_rate: float = 1000,
    order: int = 2,
) -> npt.NDArray:
    """Compute biquad filter coefficients as SOS. This function caches.

    Args:
        lowcut (float|None): Lower cutoff in Hz. Defaults to None.
        highcut (float|None): Upper cutoff in Hz. Defaults to None.
        sample_rate (float): Sampling rate in Hz. Defaults to 1000 Hz.
        order (int, optional): Filter order. Defaults to 2.

    Returns:
        npt.NDArray: SOS
    """
    nyq = sample_rate / 2
    if lowcut is not None and highcut is not None:
        freqs = [lowcut / nyq, highcut / nyq]
        btype = "bandpass"
    elif lowcut is not None:
        freqs = lowcut / nyq
        btype = "highpass"
    elif highcut is not None:
        freqs = highcut / nyq
        btype = "lowpass"
    else:
        raise ValueError("At least one of lowcut or highcut must be specified")
    sos = sps.butter(order, freqs, btype=btype, output="sos")
    return sos


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

def filter_signal(
    data: npt.NDArray,
    lowcut: float | None = None,
    highcut: float | None = None,
    sample_rate: float = 1000,
    order: int = 2,
    axis: int = -1,
    forward_backward: bool = True,
) -> npt.NDArray:
    """Apply SOS filter to signal using butterworth design and cascaded filter.

    Args:
        data (npt.NDArray): Signal
        lowcut (float|None): Lower cutoff in Hz. Defaults to None.
        highcut (float|None): Upper cutoff in Hz. Defaults to None.
        sample_rate (float): Sampling rate in Hz Defaults to 1000 Hz.
        order (int, optional): Filter order. Defaults to 2.
        forward_backward (bool, optional): Apply filter forward and backwards. Defaults to True.

    Returns:
        npt.NDArray: Filtered signal
    """
    sos = get_butter_sos(lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order)
    if forward_backward:
        return sps.sosfiltfilt(sos, data, axis=axis)
    return sps.sosfilt(sos, data, axis=axis)
