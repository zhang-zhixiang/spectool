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
    med = np.median(flux1)
    flux1 = flux1 / med
    err1 = err1 / med
    fname2 = 'template_spec.txt'
    wave2, flux2, err2 = read_spec(fname2)
    med = np.median(flux2)
    flux2 = flux2 / med
    # flux2 = flux2 - np.mean(flux2)
    # plt.plot(wave1, flux1)
    plt.plot(wave2, flux2)
    Shiftmodel = ccf.liblogccf.Shift_spec(flux2)
    # for shift in range(-51, -49, 0.37):
    # for shift in np.arange(-51, -49, 0.37):
    shift = -51
    flux3 = np.array(Shiftmodel.get_shift_spec_arr(shift))
    # print(flux2[50:60] - flux3[0:10])
    # print(flux3[:10])
    plt.plot(wave2, flux3)
    plt.show()
    t1 = time.time()
    result = ccf.find_radial_velocity2(wave1, flux1, wave2, flux2, mult=True, plot=False, ccfleft=-800, ccfright=800, velocity_resolution=1.0)
    t2 = time.time()
    print('find_radial_velocity2 run time =', t2 - t1)
    t1 = time.time()
    result = ccf.find_radial_velocity(wave1, flux1, wave2, flux2, mult=True, plot=False, ccfleft=-800, ccfright=800, velocity_resolution=1.0)
    t2 = time.time()
    print('find_radial_velocity run time =', t2 - t1)
    print(result)


if __name__ == '__main__':
    main()