# spectool

---

This is a collection of the common reduction functions of the spectroscopic data, including rebin function, smooth function, normalize function, radial velocity measurement function, et al.

#### Installation

This package only supports Linux or Mac OS, or you can install it in WSL (Windows Subsystem for Linux). You can install it by the following command:

```bash
pip install git+https://github.com/zzxihep/spectool.git
```

or you can download the source code and install it by the following command:

```bash
git clone https://github.com/zzxihep/spectool.git
cd spectool
python setup.py install
```

#### Dependencies

---

This package depends on the following packages:

- GSL [(GNU Scientific Library)](https://www.gnu.org/software/gsl/). version >= 2.7
- FFTW3
- pybind11
- PyAstronomy
If you are using Ubuntu or Debian, you can install them by the following command:

```bash
sudo apt install libgsl-dev libfftw3-dev
pip install pybind11
pip install PyAstronomy
```

If you are using Mac OS, you can install them by the following command:

```bash
brew install gsl fftw
pip install pybind11
pip install PyAstronomy
```
