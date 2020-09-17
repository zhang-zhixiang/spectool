import time
import spectool
from spectool import ccf
import numpy as np
import matplotlib.pyplot as plt


def read_spec(fname):
    data = np.loadtxt(fname)
    wave = data[:, 0]
    flux = data[:, 1]
    if data.shape[1] > 2:
        err = data[:, 2]
    else:
        err = None
    return wave, flux, err


def main():
    fname1 = 'object_spec.txt'
    wave1, flux1, err1 = read_spec(fname1)
    plt.plot(wave1, flux1)
    for ind in range(1, 10):
        start = time.time()
        spec = spectool.libspecfunc.get_normalized_spec(wave1, flux1, 30, ind)
        end = time.time()
        print('time spend = ', end - start)
        plt.plot(wave1, flux1 / spec)
    plt.show()


if __name__ == '__main__':
    main()