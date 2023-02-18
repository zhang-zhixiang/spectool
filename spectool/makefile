detected_OS := $(shell uname)

ifeq ($(detected_OS), Linux)
	FLAG = -fPIC -std=c++17 -O2
	CC = g++
else
	FLAG = -fPIC -std=c++17 -O2 -Wall -undefined dynamic_lookup
	CC = clang++
endif

suffix := $(shell python3-config --extension-suffix)
PY_CFLAGS := $(shell python3-config --includes)
PYBIND11 := $(shell python3 -m pybind11 --includes)
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
	rm $(libconvol) $(librebin) $(libccf) $(liblogccf) $(liblnspecfilter)
