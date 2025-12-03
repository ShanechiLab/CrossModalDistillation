from functools import partial

import numpy as np
import torch
from einops import rearrange
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirnotch, resample, welch

from cross_modal_distillation.utility.utils import init_logger

std_logger = init_logger("Preprocessing")


def notch_filter(data, fs, freq=60, quality_factor=30):
    """
    quality factor is defined as freq / (bandwith to remove around freq)
    so, if you want to remove frequencies +-2 Hz of notch freq, choose it 2Hz,
    thus, 60/2=30.
    """
    filtered_data = data
    nyquist = fs / 2  # Nyquist frequency

    # Iterate through harmonics until exceeding the Nyquist frequency
    harmonic_freq = freq
    while harmonic_freq < nyquist:
        b, a = iirnotch(harmonic_freq, quality_factor, fs)
        filtered_data = filtfilt(b, a, filtered_data, axis=0)
        harmonic_freq += freq  # Move to the next harmonic
    if not isinstance(filtered_data, torch.Tensor):
        filtered_data = torch.tensor(filtered_data.copy(), dtype=torch.float32)
    return filtered_data


def downsample(data, original_fs=None, target_fs=None, t=None, num_samples=None):
    if num_samples is None:
        if original_fs is None or target_fs is None:
            raise ValueError(
                "'num_samples' is None, and 'original_fs' or 'target_fs' is None, cannot do downsampling."
            )

        if target_fs > original_fs:
            raise ValueError(
                "Target sampling rate must be less than or equal to the original sampling rate."
            )
    else:
        if original_fs is not None and target_fs is not None:
            print(
                "Downsampling via downsample fn: 'num_samples' is passed, 'original_fs' and 'target_fs' are discarded."
            )

    if num_samples is None:
        num_samples = int(len(data) * target_fs / original_fs)

    ds_data = resample(data, num_samples)
    if not isinstance(ds_data, torch.Tensor):
        ds_data = torch.tensor(ds_data.copy(), dtype=torch.float32)

    if t is not None:
        t0, tend = t[0], t[-1]
        num_steps = ds_data.shape[0]
        ds_t = torch.linspace(t0, tend, num_steps)
        return ds_data, ds_t
    else:
        return ds_data, None


def downsample_in_chunks(data, original_fs, target_fs, chunk_size=1e6, t=None):
    if target_fs > original_fs:
        raise ValueError(
            "Target sampling rate must be less than or equal to the original sampling rate."
        )

    # Create an empty list to store downsampled chunks
    downsampled_chunks = []

    # Process each chunk separately
    for start_idx in range(0, len(data), int(chunk_size)):
        end_idx = int(min(start_idx + chunk_size, len(data)))
        chunk = data[start_idx:end_idx]

        # Downsample the chunk
        num_samples = int(len(chunk) * target_fs / original_fs)
        downsampled_chunk = resample(chunk, num_samples)
        downsampled_chunks.append(downsampled_chunk)

    # Concatenate all the downsampled chunks into a single array
    ds_data = np.concatenate(downsampled_chunks, axis=0)
    ds_data = torch.tensor(ds_data.copy(), dtype=torch.float32)

    if t is not None:
        t0, tend = t[0], t[-1]
        num_steps = ds_data.shape[0]
        ds_t = torch.linspace(t0, tend, num_steps)
        return ds_data, ds_t
    else:
        return ds_data, None


def highpass_filter(data, fs, cutoff=1, order=2):
    b, a = butter(order, cutoff / (fs / 2), btype="highpass")
    filtered_data = filtfilt(b, a, data, axis=0)
    filtered_data = torch.tensor(filtered_data.copy(), dtype=torch.float32)
    return filtered_data


def lowpass_filter(data, fs, cutoff=200, order=2):
    b, a = butter(order, cutoff / (fs / 2), btype="lowpass")
    filtered_data = filtfilt(b, a, data, axis=0)
    filtered_data = torch.tensor(filtered_data.copy(), dtype=torch.float32)
    return filtered_data


def common_average_reference(data):
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    data_car = data - np.mean(data, axis=1, keepdims=True)
    data_car = torch.tensor(data_car.copy(), dtype=torch.float32)
    return data_car


def align_signal(data, t, align_t):
    b = None
    if len(data.shape) == 3:
        t_, c, b = data.shape
        data = data.reshape(t_, -1)
    else:
        t_, c = data.shape

    d = data.shape[-1]
    aligned_data = []
    for i in range(d):
        interpolator = interp1d(t, data[:, i], kind="linear", fill_value="extrapolate")
        aligned_data.append(interpolator(align_t))
    aligned_data = np.stack(aligned_data, axis=-1)
    aligned_data = torch.tensor(aligned_data.copy(), dtype=torch.float32)

    if b is not None:
        aligned_data = aligned_data.reshape(-1, c, b)
    return aligned_data
