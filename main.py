import scipy.stats as stats
import numpy as np
from scipy.stats import levy_stable
from scipy.signal import lfilter
from numpy.fft import fft, fftshift, fftfreq
import matplotlib.pylab as plt

import padasip as pa

from adaptive_mmse import adaptive_wiener
from correlation import xcorr

plt.style.use('ggplot')  # nicer plots

N0 = 1.5
N = 10000
t = np.arange(N)
# alpha noise
alpha = 1.8
r = levy_stable.rvs(alpha, 1.0, 0, size=N)
# correlation
Rbiased, lags = xcorr(r)
# PSD
Yf = fft(r)
Py = 1 / N * abs(Yf) ** 2
f = fftfreq(N)
f = np.linspace(-0.5, 0.5, N)

# given plant with H(z) = (1-z^-10)/(1-z^-1)
yw = lfilter([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1], [1, -1], r)
# white noise, gaussian with sigma^2 = N0/2
XGauss = stats.norm(loc=0, scale=np.sqrt(N0 / 2))
s = XGauss.rvs(size=N)
# add white noise
y = yw + s

# correlation
Rybiased, ylags = xcorr(y)
# PSD
Yyf = fft(y)
Pyy = 1 / N * abs(Yyf) ** 2
fy = np.linspace(-0.5, 0.5, N)

# plot input/output
fig, ax = plt.subplots(6, 1, figsize=(10, 15))
ax[0].plot(t, r)
ax[0].set_title("Alpha stable noise, alpha=1.6")
ax[0].set_xlabel("Delay")
ax[0].axis('tight')  # Tight layout of the axis

# biased correlation
ax[1].plot(lags, Rbiased)
ax[1].set_title("Correlation function")
ax[1].set_xlabel("Delay")
ax[1].axis('tight')  # Tight layout of the axis

# PSD
ax[2].plot(f, fftshift(Py), alpha=0.65, label="Periodogram")
ax[2].set_title("Input PSD")
ax[2].axis('tight')  # Tight layout of the axis

ax[3].plot(t, y)
ax[3].set_title("Output with Gaussian White noise")
ax[3].set_xlabel("Delay")
ax[3].axis('tight')  # Tight layout of the axis

# biased correlation
ax[4].plot(ylags, Rybiased)
ax[4].set_title("Correlation function")
ax[4].set_xlabel("Delay")
ax[4].axis('tight')  # Tight layout of the axis

# PSD
ax[5].plot(fy, fftshift(Pyy), alpha=0.65, label="Periodogram")
ax[5].set_title("Output PSD")
ax[5].axis('tight')  # Tight layout of the axis

plt.legend()
fig.tight_layout()
plt.show()

window = 100
qmax = 15  # max value for q
w, mmse = adaptive_wiener(15, window, r, y)

print("MMSE: ", mmse)

plt.plot(range(0, qmax), mmse)
plt.xlabel("Order of the filter")
plt.ylabel("MMSE")
plt.title("MMSE as a function of the length of the identification filter")
plt.show()

zy = lfilter(w, [1], r)

# show results
plt.figure(figsize=(12.5, 9))
plt.subplot(221)
plt.title("Target Signal")
plt.xlabel("Number of iteration [-]")
plt.plot(y, "b", label="d - target")
plt.xlim(0, N)
plt.legend()

plt.subplot(222)
plt.title("MMSE Adaptation")
plt.xlabel("Number of iteration [-]")
plt.plot(yw, "b", label="d - target")
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
plt.plot(pa.misc.logSE(yw, zy), "r", label="Squared error [dB]")
plt.legend()
plt.xlim(0, N)
plt.tight_layout()

# identification
window = 30
x = pa.input_from_history(r, n=window)
f = pa.filters.FilterNLMS(mu=0.9, n=window)
y = y[:-(window - 1)]
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

err = abs(pa.misc.logSE(yw, zy).min())
print("ERROR MSME: ", err)
err_lms = abs(pa.misc.logSE(e).min())
print("ERROR LMS: ", err_lms)
