import scipy.stats as stats
import numpy as np
from scipy.stats import levy_stable
from scipy.signal import lfilter
from numpy.fft import fft, fftshift, fftfreq
import matplotlib.pylab as plt
from scipy.io import wavfile
import scipy
import scipy.fftpack

import padasip as pa

from adaptive_mmse import adaptive_wiener
from correlation import xcorr

plt.style.use('ggplot')  # nicer plots

fs_rate, signal = wavfile.read("sound.WAV")
print ("Frequency sampling", fs_rate)
l_audio = len(signal.shape)
print ("Channels", l_audio)
if l_audio == 2:
    signal = signal.sum(axis=1) / 2
N = signal.shape[0]
print ("Complete Samplings N", N)
secs = N / float(fs_rate)
print ("secs", secs)
Ts = 1.0/fs_rate # sampling interval in time
print ("Timestep between samples Ts", Ts)
t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
FFT = abs(scipy.fft(signal))

freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
fft_freqs = np.array(freqs)

plt.subplot(211)
p1 = plt.plot(t, signal, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(312)
p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')
plt.show()

# correlation
Rybiased, ylags = xcorr(signal)
# PSD
Yyf = fft(signal)
Pyy = 1 / N * abs(Yyf) ** 2
fy = np.linspace(-0.5, 0.5, N)

# plot input/output
fig, ax = plt.subplots(2, 1, figsize=(10, 15))
# biased correlation
ax[0].plot(ylags, Rybiased)
ax[0].set_title("Correlation function")
ax[0].set_xlabel("Delay")
ax[0].axis('tight')  # Tight layout of the axis


# PSD
ax[1].plot(fy, fftshift(Pyy), alpha=0.65, label="Periodogram")
ax[1].set_title("Input PSD")
ax[1].axis('tight')  # Tight layout of the axis

plt.show()

#max power
#pwr = Pyy.max()/fs_rate

# Gaussian white with same power
N0 = 0.1
XGauss = stats.norm(loc=0, scale=np.sqrt(N0 / 2))
x = XGauss.rvs(size=N)

window = 100
qmax = 15  # max value for q
w, mmse = adaptive_wiener(qmax, window, x, signal)

print("MMSE: ", mmse)

plt.plot(range(0, qmax), mmse)
plt.xlabel("Order of the filter")
plt.ylabel("MMSE")
plt.title("MMSE as a function of the length of the identification filter")
plt.show()

zy = window*lfilter(w, [1], x)

# show results
plt.figure(figsize=(12.5, 9))
plt.subplot(221)
plt.title("Target Signal")
plt.xlabel("Number of iteration [-]")
plt.plot(signal, "b", label="d - target")
plt.xlim(0, N)
plt.legend()

plt.subplot(222)
plt.title("MMSE Adaptation")
plt.xlabel("Number of iteration [-]")
plt.plot(signal, "b", label="d - target")
plt.plot(zy, "g", label="y - output")
plt.xlim(0, N)
plt.legend()

plt.subplot(223)
plt.title("MMSE Predicted Signal")
plt.xlabel("Number of iteration [-]")
plt.plot(zy, "g", label="y - output")
plt.xlim(0, N)
plt.legend()

plt.subplot(224)
plt.title("MMSE Filter error")
plt.xlabel("Number of iteration [-]")
plt.plot(pa.misc.logSE(signal, zy), "r", label="Squared error [dB]")
plt.legend()
plt.xlim(0, N)
plt.tight_layout()

plt.show()


# identification
window = 500
x = pa.input_from_history(x, n=window)
f = pa.filters.FilterNLMS(mu=0.9, n=window)
y = signal[:-(window - 1)]
y1, e, w = f.run(y, x)

# show results
plt.figure(figsize=(12.5, 9))
plt.subplot(221)
plt.title("Target Signal")
plt.xlabel("Number of iteration [-]")
plt.plot(y, "b", label="d - target")
plt.xlim(0, N)
plt.legend()

plt.subplot(222)
plt.title("LMS Adaptation")
plt.xlabel("Number of iteration [-]")
plt.plot(y, "b", label="d - target")
plt.plot(y1, "g", label="y - output")
plt.xlim(0, N)
plt.legend()

plt.subplot(223)
plt.title("LMS Predicted Signal")
plt.xlabel("Number of iteration [-]")
plt.plot(y1, "g", label="y - output")
plt.xlim(0, N)
plt.legend()

plt.subplot(224)
plt.title("Filter error")
plt.xlabel("Number of iteration [-]")
plt.plot(pa.misc.logSE(e), "r", label="Squared error [dB]")
plt.legend()
plt.xlim(0, N)
plt.tight_layout()
plt.show()
print("And the resulting coefficients are: {}".format(w[-1]))