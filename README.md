# spectool

spectool is a collection of common reduction functions for spectroscopic data, providing tools for rebinning, smoothing, normalization, radial velocity measurement, and more.

## Features

- Rebinning & Interpolation: Functions for resampling spectra to a new wavelength grid using different interpolation methods.
- Spectral Smoothing & Filtering: Various smoothing methods, including Gaussian and rotation kernel filters, for both wavelength and velocity space.
- Spectral Normalization: Normalization of spectral data using median filter and polynomial fitting techniques.
- Spectral Matching & Fitting: Tools for matching and scaling spectra to a reference spectrum using polynomial fitting and other methods.
- Radial Velocity Measurement: Functions for calculating radial velocities from spectroscopic data.

## Installation

This package supports only Linux or Mac OS, or you can install it in WSL (Windows Subsystem for Linux). To install, run:

```bash
pip install git+https://github.com/zhang-zhixiang/spectool.git
```

Alternatively, you can download the source code and install it manually:

```bash
git clone https://github.com/zhang-zhixiang/spectool.git
cd spectool
python setup.py install
```

## Dependencies

This package depends on the following packages:

- [GSL](https://www.gnu.org/software/gsl/) (version >= 2.7)
- FFTW3
- pybind11
- PyAstronomy

For Ubuntu/Debian, you can install them using:

```bash
sudo apt install libgsl-dev libfftw3-dev
pip install pybind11
pip install PyAstronomy
```

For Mac OS, install them using:

```bash
brew install gsl fftw
pip install pybind11
pip install PyAstronomy
```
