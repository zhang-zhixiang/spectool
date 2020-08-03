import numpy as np


def normalize_wave(wave):
    """normalize the wavelength range to [-1, ..., 1]

    Args:
        wave (numpy.ndarray(float64)): wavelength data

    Returns:
        numpy.ndarray(float64): wavelength after the normalization
    """
    start, end = wave[0], wave[-1]
    scale = 1 / (end - start) * 1.9999
    med = (end + start) / 2
    new_wave = (wave - med) * scale
    return new_wave
