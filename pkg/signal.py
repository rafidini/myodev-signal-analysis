import biosignalsnotebooks as bsnb # signal handling
import typing # functions
import numpy as np # vector operations
import pandas as pd # data analysis/manipulation
import matplotlib.pyplot as plt

# Functions - File
def get_file_content(filepath: str) -> typing.Union[None, pd.DataFrame]:
    """
    Get the content of a given file from its path.

    @param filepath (str) filepath
    @return (str) content of the file in string
    """
    content = None
    try:
        # Get the content
        values = np.loadtxt(filepath)
        
        # Transform to pandas.DataFrame
        columns = ["nSeq", "I1", "I2", "O1", "O2", "A1"]
        content = pd.DataFrame(values, columns=columns)
    except Exception as e:
        raise e
    
    return content

# Functions - Signals
def apply_tkeo_operator(signal: np.array) -> np.array:
    """
    Apply the TKEO operator to a given signal.

    @param signal (numpy.array) the signal
    @return (numpy.array) TKEO processed signal
    """
    tkeo = []
    for i in range(0, len(signal)):
        if i == 0 or i == len(signal) - 1:
            tkeo.append(signal[i])
        else:
            tkeo.append((signal[i] ** 2) - (signal[i + 1] * signal[i - 1]))
    
    return np.array(tkeo)

def apply_smoothing(signal: np.array, sampling_rate: int, smooth_level_perc: int) -> np.array:
    """
    Smooth a given signal.

    @param signal (numpy.array) the signal
    @return (numpy.array) smoothed signal
    """
    # Compute parameters
    smooth_level = int((smooth_level_perc / 100) * sampling_rate)
    rect_signal = np.abs(signal)
    smoothed1 = []
    smoothed2 = []

    # 1st smoothing
    smoothed1 = bsnb.aux_functions._moving_average(rect_signal, sampling_rate / 10)

    # 2nd smoothing
    for i in range(0, len(smoothed1)):
        if smooth_level < i < len(smoothed1) - smooth_level:
            smoothed2.append(np.mean(smoothed1[i - smooth_level:i + smooth_level]))
        else:
            smoothed2.append(0)

    return np.array(smoothed2)

def get_duration(signal: np.array, smooth_level:int=20, threshold_level:int=10):
    burst_begin, burst_end = bsnb.detect_emg_activations(signal, 1000, smooth_level=smooth_level, threshold_level=threshold_level, time_units=True)[:2]
    burst_duration = burst_end - burst_begin
    max_duration = np.max(burst_duration)
    min_duration = np.max(burst_duration)
    avg_duration = np.max(burst_duration)
    std_duration = np.std(burst_duration)
    return dict(avg=avg_duration, max=max_duration, min=min_duration, std=std_duration, no_contraction=len(burst_begin))

def plot_signal(signal: np.array, smooth_level:int=20, threshold_level:int=10):
    #tkeo_signal = apply_tkeo_operator(signal)
    smoothed_signal = apply_smoothing(signal.values, 1000, smooth_level_perc=smooth_level)
    duration = get_duration(signal, smooth_level=smooth_level, threshold_level=threshold_level)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(np.arange(signal.shape[0]), signal, label='signal', color='lightblue', alpha=.8)
    #ax.plot(np.arange(tkeo_signal.shape[0]), tkeo_signal, label='tkeo signal', color='purple', alpha=.75)
    ax.plot(np.arange(smoothed_signal.shape[0]), smoothed_signal, label='smoothed signal', color='black', alpha=.5)
    ax.set_title(f"N={duration['no_contraction']}, Duration~{round(duration['avg'], 3)}")
    ax.legend()
    
    return fig, ax
