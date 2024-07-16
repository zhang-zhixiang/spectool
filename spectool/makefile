detected_OS := $(shell uname)

ifeq ($(detected_OS), Linux)
	FLAG = -fPIC -std=c++17 -O2
	CC = g++
else
	FLAG = -fPIC -std=c++17 -O2 -Wall -undefined dynamic_lookup
	CC = clang++
endif

PYTHON_VERSION ?= python3
PYTHON_CONFIG := $(PYTHON_VERSION)-config
suffix := $(shell $(PYTHON_CONFIG) --extension-suffix)
PY_CFLAGS := $(shell $(PYTHON_CONFIG) --includes)
PYBIND11 := $(shell $(PYTHON_VERSION) -m pybind11 --includes)
GSL := $(shell pkg-config --cflags --libs gsl)
FFTW := $(shell pkg-config --cflags --libs fftw3)
SHARE = -shared

librebin = rebin$(suffix)
libconvol = convol$(suffix)
libccf = libccf$(suffix)
liblogccf = liblogccf$(suffix)
liblnspecfilter = lnspecfilter$(suffix)

default : $(libconvol) $(librebin) $(libccf) $(liblogccf) $(liblnspecfilter)

$(libconvol) : convol.cpp types.cpp
	$(CC) convol.cpp types.cpp -o $(libconvol) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11) $(GSL)

$(librebin) : rebin.cpp types.cpp
	$(CC) rebin.cpp types.cpp -o $(librebin) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11) $(GSL)

$(libccf) : iccf.cpp types.cpp
	$(CC) iccf.cpp types.cpp -o $(libccf) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11)

$(liblogccf) : logccf.cpp types.cpp
	$(CC) logccf.cpp types.cpp -o $(liblogccf) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11) $(FFTW) $(GSL)

$(liblnspecfilter) : ln_spec_filter.cpp types.cpp
	$(CC) ln_spec_filter.cpp types.cpp -o $(liblnspecfilter) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11)

clean :
	rm -f *$(suffix) *.so *.dylib *.dll