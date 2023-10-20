import numpy as np
import torch

def get_cosine_peak(length):
    # peak_length = int(length/3)
    peak_length = int(length * 0.6)  # 1117
    N = length - peak_length
    time = np.arange(-np.pi, np.pi, np.pi * 2 / N)
    cos_amplitude = (np.cos(time) + 1.) / 2
    middle_peak = np.ones(peak_length)
    w1 = np.concatenate([cos_amplitude[:int(N / 2)], middle_peak, cos_amplitude[int(N / 2):]])
    return w1


def get_cosine(length):
    time = np.arange(-np.pi, np.pi, np.pi * 2 / length)
    w1 = (np.cos(time) + 1.) / 2
    return w1


def zero_both_side_mask(length, zero_length_ratio=0.1):
    mask = np.ones(length)
    zero_length = 1
    mask[:zero_length] = 0.5
    mask[zero_length:] = 0.5
    return mask


def get_mix_weight(length=30, mode='replace'):
    """
    return different mask for mix enc_output
    """
    w1 = np.ones(length)
    if mode == 'replace':
        pass
    elif mode == '0505':
        w1 = w1 * 0.5
    elif mode == '0703':
        w1 = w1 * 0.9
    elif mode == '0901':
        w1 = w1 * 0.9
    elif mode == '1000':
        w1 = w1 * 1.0

    elif mode == 'cosine-07':
        w1 = get_cosine(length) * 0.7
    elif mode == 'cosine-09':
        w1 = get_cosine(length) * 0.9
    elif mode == 'cosine-10':
        w1 = get_cosine(length) * 1.0

    elif mode == 'cosinepeak-07':
        w1 = get_cosine_peak(length) * 0.7
    elif mode == 'cosinepeak-08':
        w1 = get_cosine_peak(length) * 0.8
    elif mode == 'cosinepeak-09':
        w1 = get_cosine_peak(length) * 0.9
    elif mode == 'cosinepeak-10':
        w1 = get_cosine_peak(length) * 1.0
    else:
        assert False
    w1 = torch.from_numpy(w1)  # .unsqueeze(0).unsqueeze(-1)
    return w1
