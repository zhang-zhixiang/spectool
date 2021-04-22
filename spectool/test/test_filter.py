import sys
import numpy as np
import matplotlib.pyplot as plt
import spectool
sys.path.append('..')
import convol
import tqdm


def main():
    wave = np.arange(3000, 8000, 1.1)
    # flux = np.ones(wave.shape)
    flux = np.zeros(wave.shape)
    flux[800] = 123
    flux[2500] = 99
    velocity = np.arange(-4000, 4000, 10.0)
    profile = np.zeros(velocity.shape)
    arg = np.where((velocity>-2000) & (velocity<2000))
    profile[arg] = 1.0
    outflux = convol.filter_use_given_profile(wave, flux, velocity, profile)
    plt.step(wave, flux)
    plt.step(wave, outflux)
    plt.show()


def main3():
    wave = np.arange(3000, 6000, 1.2)
    velocity = np.arange(-4000, 4000, 100.0)
    profile = np.zeros(velocity.shape)
    arg = np.where((velocity>-2000) & (velocity<2000))
    profile[arg] = 1.0
    sumflux = np.zeros(wave.shape)
    for ind in tqdm.tqdm(range(len(wave))):
        flux = np.zeros(wave.shape)
        flux[ind] = 1.0
        outflux = convol.filter_use_given_profile(wave, flux, velocity, profile)
        sumflux = sumflux + outflux
    # flux = np.ones(wave.shape)
    # flux[800] = 123
    # flux[2500] = 99

    plt.step(wave, flux)
    plt.step(wave, sumflux)
    plt.show()


def main2():
    # wave = np.arange(3000, 8000, 1.2)
    # flux = np.ones(wave.shape)
    # flux[800] = 123
    # flux[2500] = 99
    data = np.loadtxt('template_spec.txt')
    wave = data[:, 0]
    flux = data[:, 1]
    velocity = np.arange(-4000, 4000, 1.0)
    profile = np.zeros(velocity.shape)
    profile[3500:4500] = 1.0
    outflux = convol.filter_use_given_profile(wave, flux, velocity, profile)
    out_flux2 = spectool.spec_filter.gaussian_filter(wave, flux, 500)
    plt.plot(velocity, profile)
    plt.figure()
    plt.step(wave, flux)
    plt.step(wave, outflux, label='rect')
    plt.step(wave, out_flux2, label='gaussian')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # main()
    # main2()
    main3()