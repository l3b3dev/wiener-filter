import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import inv

from correlation import xcorr


def adaptive_wiener(filter_size, filter_window, xin, out):
    u = xin[0:filter_window]
    d = out[0:filter_window]

    se = np.zeros(filter_size)  # initialize the vector of errors
    for q in range(0, filter_size):
        c = xcorr(u, u, maxlags=q)[0][q::]  # correlation vector
        Ruu = toeplitz(c)
        Rdu = xcorr(d, u, maxlags=q)[0][q::]
        w = inv(Ruu).dot(Rdu)
        # Minimum error
        sigma2d = np.mean(d ** 2)
        se[q] = sigma2d - w.dot(Rdu)

    return w, se