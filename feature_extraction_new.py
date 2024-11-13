import numpy as np
import math
import pywt

def selected_features(x, fs, frame, eps):
    variance, rms, log_detector, wl, dasdv, zc, mnp, tot, mnf, mdf = [], [], [], [], [], [], [], [], [], []

    # time_features
    th = np.mean(x) + 3 * np.std(x)
    variance.append(np.var(x))
    rms.append(np.sqrt(np.mean(x ** 2)))
    log_detector.append(np.exp(np.sum(np.log10(np.absolute(x) + eps)) / frame))
    wl.append(np.sum(abs(np.diff(x))))  # Wavelength
    dasdv.append(math.sqrt((1 / (frame - 1)) * np.sum((np.diff(x)) ** 2)))  # Difference absolute standard deviation value
    zc.append(zcruce(x, th))  # Zero-Crossing

    # frequency_features
    frequency, power = spectrum(x, fs)
    mnp.append(np.sum(power) / len(power))  # Mean power
    tot.append(np.sum(power))  # Total power
    mnf.append(mean_freq(frequency, power, eps))  # Mean frequency
    mdf.append(median_freq(frequency, power, eps))  # Median frequency

    features_matrix = np.column_stack((variance, rms, log_detector, wl, dasdv, zc, mnp, tot, mnf, mdf))
    
    return features_matrix


def time_features(x, frame, eps):
    variance = []
    rms = []
    iemg = []
    mav = []
    log_detector = []
    wl = []
    aac = []
    dasdv = []
    zc = []
    wamp = []
    myop = []


    th = np.mean(x) + 3 * np.std(x)


    variance.append(np.var(x))
    rms.append(np.sqrt(np.mean(x ** 2)))
    iemg.append(np.sum(abs(x)))  # Integral
    mav.append(np.sum(np.absolute(x)) / frame)  # Mean Absolute Value
    log_detector.append(np.exp(np.sum(np.log10(np.absolute(x) + eps)) / frame))
    wl.append(np.sum(abs(np.diff(x))))  # Wavelength
    aac.append(np.sum(abs(np.diff(x))) / frame)  # Average Amplitude Change
    dasdv.append(
        math.sqrt((1 / (frame - 1)) * np.sum((np.diff(x)) ** 2)))  # Difference absolute standard deviation value
    zc.append(zcruce(x, th))  # Zero-Crossing
    wamp.append(wilson_amplitude(x, th))  # Willison amplitude
    myop.append(myopulse(x, th))  # Myopulse percentage rate

    time_features_matrix = np.column_stack((variance, rms, iemg, mav, log_detector, wl, aac, dasdv, zc, wamp, myop))

    return time_features_matrix


def frequency_features(x, fs, frame, eps):
    """
    Compute frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param fs: sampling frequency of the signal.
    :param frame: sliding window size
    :param step: sliding window step size

    :return: frequency_features_matrix: narray matrix with the frequency features stacked by columns.
    """

    fr = []
    mnp = []
    tot = []
    mnf = []
    mdf = []
    pkf = []

    
    frequency, power = spectrum(x, fs)

    fr.append(frequency_ratio(frequency, power, eps))  # Frequency ratio
    mnp.append(np.sum(power) / len(power))  # Mean power
    tot.append(np.sum(power))  # Total power
    mnf.append(mean_freq(frequency, power, eps))  # Mean frequency
    mdf.append(median_freq(frequency, power, eps))  # Median frequency
    pkf.append(frequency[power.argmax()])  # Peak frequency

    frequency_features_matrix = np.column_stack((fr, mnp, tot, mnf, mdf, pkf))

    return frequency_features_matrix


def time_frequency_features(x, frame, eps):
    h_wavelet = []

    E_a, E = wavelet_energy(x, 'db2', 3, eps)
    E.insert(0, E_a)
    E = np.asarray(E) / 100

    h_wavelet.append(-np.sum(E * np.log2(E + eps)))

    return h_wavelet


def time_features_estimation(signal, frame, step, eps):
    """
    Compute time features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size.
    :param step: sliding window step size.

    :return: time_features_matrix: narray matrix with the time features stacked by columns.
    """

    variance = []
    rms = []
    iemg = []
    mav = []
    log_detector = []
    wl = []
    aac = []
    dasdv = []
    zc = []
    wamp = []
    myop = []

    th = np.mean(signal) + 3 * np.std(signal)

    for i in range(frame, signal.size+1, step):
        x = signal[i - frame:i]

        variance.append(np.var(x))
        rms.append(np.sqrt(np.mean(x ** 2)))
        iemg.append(np.sum(abs(x)))  # Integral
        mav.append(np.sum(np.absolute(x)) / frame)  # Mean Absolute Value
        log_detector.append(np.exp(np.sum(np.log10(np.absolute(x) + eps)) / frame))
        wl.append(np.sum(abs(np.diff(x))))  # Wavelength
        aac.append(np.sum(abs(np.diff(x))) / frame)  # Average Amplitude Change
        dasdv.append(
            math.sqrt((1 / (frame - 1)) * np.sum((np.diff(x)) ** 2)))  # Difference absolute standard deviation value
        zc.append(zcruce(x, th))  # Zero-Crossing
        wamp.append(wilson_amplitude(x, th))  # Willison amplitude
        myop.append(myopulse(x, th))  # Myopulse percentage rate

    time_features_matrix = np.column_stack((variance, rms, iemg, mav, log_detector, wl, aac, dasdv, zc, wamp, myop))
    return time_features_matrix



def frequency_features_estimation(signal, fs, frame, step, eps):
    """
    Compute frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param fs: sampling frequency of the signal.
    :param frame: sliding window size
    :param step: sliding window step size

    :return: frequency_features_matrix: narray matrix with the frequency features stacked by columns.
    """

    fr = []
    mnp = []
    tot = []
    mnf = []
    mdf = []
    pkf = []

    for i in range(frame, signal.size+1, step):
        x = signal[i - frame:i]
        frequency, power = spectrum(x, fs)

        fr.append(frequency_ratio(frequency, power, eps))  # Frequency ratio
        mnp.append(np.sum(power) / len(power))  # Mean power
        tot.append(np.sum(power))  # Total power
        mnf.append(mean_freq(frequency, power, eps))  # Mean frequency
        mdf.append(median_freq(frequency, power, eps))  # Median frequency
        pkf.append(frequency[power.argmax()])  # Peak frequency

    frequency_features_matrix = np.column_stack((fr, mnp, tot, mnf, mdf, pkf))

    return frequency_features_matrix


def time_frequency_features_estimation(signal, frame, step, eps):
    """
    Compute time-frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size
    :param step: sliding window step size

    :return: h_wavelet: list
    """
    h_wavelet = []

    for i in range(frame, signal.size+1, step):
        x = signal[i - frame:i]

        E_a, E = wavelet_energy(x, 'db2', 3, eps)
        E.insert(0, E_a)
        E = np.asarray(E) / 100

        h_wavelet.append(-np.sum(E * np.log2(E + eps)))

    return h_wavelet


def wilson_amplitude(signal, th):
    x = abs(np.diff(signal))
    umbral = x >= th
    return np.sum(umbral)


def myopulse(signal, th):
    umbral = signal >= th
    return np.sum(umbral) / len(signal)


def spectrum(signal, fs):
    m = len(signal)
    n = next_power_of_2(m)
    y = np.fft.fft(signal, n)
    yh = y[0:int(n / 2 - 1)]
    fh = (fs / n) * np.arange(0, n / 2 - 1, 1)
    power = np.real(yh * np.conj(yh) / n)

    return fh, power


def frequency_ratio(frequency, power, eps):
    power_low = power[(frequency >= 30) & (frequency <= 250)]
    power_high = power[(frequency > 250) & (frequency <= 500)]
    ULC = np.sum(power_low)
    UHC = np.sum(power_high)

    return ULC / (UHC + eps)


def shannon(x):
    N = len(x)
    nb = 19
    hist, bin_edges = np.histogram(x, bins=nb)
    counts = hist / N
    nz = np.nonzero(counts)

    return np.sum(counts[nz] * np.log(counts[nz]) / np.log(2))


def zcruce(X, th):
    th = 0
    cruce = 0
    for cont in range(len(X) - 1):
        can = X[cont] * X[cont + 1]
        can2 = abs(X[cont] - X[cont + 1])
        if can < 0 and can2 > th:
            cruce = cruce + 1
    return cruce


def mean_freq(frequency, power, eps):
    num = 0
    den = 0
    for i in range(int(len(power) / 2)):
        num += frequency[i] * power[i]
        den += power[i]

    return num / (den + eps)


def median_freq(frequency, power, eps):
    power_total = np.sum(power) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0

    while abs(errel) > tol:
        temp += power[i]
        errel = (power_total - temp) / (power_total + eps)
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return frequency[i]


def wavelet_energy(x, mother, nivel, eps):
    coeffs = pywt.wavedecn(x, wavelet=mother, level=nivel)
    arr, _ = pywt.coeffs_to_array(coeffs)
    Et = np.sum(arr ** 2)
    cA = coeffs[0]
    Ea = 100 * np.sum(cA ** 2) / (Et + eps)
    Ed = []

    for k in range(1, len(coeffs)):
        cD = list(coeffs[k].values())
        cD = np.asarray(cD)
        Ed.append(100 * np.sum(cD ** 2) / (Et + eps))

    return Ea, Ed


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def med_freq(f, P):
    Ptot = np.sum(P) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0

    while abs(errel) > tol:
        temp += P[i]
        errel = (Ptot - temp) / Ptot
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return f[i]