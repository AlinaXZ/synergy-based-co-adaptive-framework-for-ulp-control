import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# notch filter

def notch_filter(x, samplerate, plot=False):
    x = x - np.mean(x)

    high_cutoff_notch = 59 / (samplerate / 2)
    low_cutoff_notch = 61 / (samplerate / 2)

    [b, a] = signal.butter(4, [high_cutoff_notch, low_cutoff_notch], btype='stop')

    x_filt = signal.filtfilt(b, a, x.T)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt.T, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt

# band pass filter

def bp_filter(x, low_f, high_f, samplerate, plot=False):
    x = x - np.mean(x)

    low_cutoff_bp = low_f / (samplerate / 2)
    high_cutoff_bp = high_f / (samplerate / 2)

    [b, a] = signal.butter(5, [low_cutoff_bp, high_cutoff_bp], btype='bandpass')

    x_filt = signal.filtfilt(b, a, x)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt

# create default label list
def num_sequence(n):
    seq = []
    for i in range(n):
        seq.append(str(i+1))
    return seq


# plot
def plot_signal_multi(data_list:list, subplot, labels=None, fig_size=None):
    """
    Subplot each channel of the input signal in a signal list
        data_list: signal data list, for one signal, data = [signal], 
            type: list of Dataframe or Array, shape: [data1, data2, ...]
            data shape: (length of signal, number of channels)
        subplot: (row, column) number of subplot figures, type: tuple (int, int)
        labels: list of data labels
        fig_size: figure size, example: fig_size=(16,8)
    """

    # 补全labels
    if labels == None:
        labels = num_sequence(len(data_list))
    elif len(labels) < len(data_list):
        n_miss = len(data_list)-len(labels)
        labels = labels + ["missing label"] * n_miss

    row, col = subplot
    if fig_size == None:
        fig_size = (col*3, row*2)
    fig, axes = plt.subplots(row, col, figsize=fig_size, sharex=True, sharey=True)
    
    channels = data_list[0].shape[1]

    # plot
    if row == 1 or col == 1:
        if row*col == 1:
            [axes.plot(data[:,0], label=labels[label_i]) for label_i, data in enumerate(data_list)]
        else:
            for i in range(row*col):
                if i < channels:
                    [axes[i].plot(data[:,i], label=labels[label_i]) for label_i, data in enumerate(data_list)]
                else:
                    fig.delaxes(axes[i])

    else:
        for i in range(row):
            for j in range(col):
                if (i*col+j) < channels:
                    [axes[i][j].plot(data[:,i*col+j], label=labels[label_i]) for label_i, data in enumerate(data_list)]
                else:
                    fig.delaxes(axes[i][j])

    if len(data_list) != 1:
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc = 'upper center')


def plot_signal(data_list:list, samp_freq, labels=None, fig_size=None):

    # 补全labels
    if labels == None:
        labels = num_sequence(len(data_list))
    elif len(labels) < len(data_list):
        n_miss = len(data_list)-len(labels)
        labels = labels + ["missing label"] * n_miss

    if fig_size:
        plt.figure(figsize=fig_size)

    t = np.arange(0, len(data_list[0]) / samp_freq, 1 / samp_freq)
    [plt.plot(t, sig, label=label) for (sig, label) in zip(data_list, labels)]

    plt.legend(loc = 'lower left')
    plt.autoscale(tight=True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.show()