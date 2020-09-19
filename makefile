detected_OS := $(shell uname)

ifeq ($(detected_OS), Linux)
	FLAG = -fPIC -std=c++17 -O3
	CC = g++
else
	FLAG = -fPIC -std=c++17 -O3 -Wall -undefined dynamic_lookup
	CC = clang++
endif

suffix := $(shell python3-config --extension-suffix)
PY_CFLAGS := $(shell python3-config --includes)
PYBIND11 := $(shell python3 -m pybind11 --includes)
GSL := $(shell pkg-config --libs gsl)
FFTW := $(shell pkg-config --libs fftw3)
SHARE = -shared

librebin = rebin$(suffix)
libconvol = convol$(suffix)
libccf = libccf$(suffix)
liblogccf = liblogccf$(suffix)
libspecfunc = libspecfunc$(suffix)

default : $(libconvol) $(librebin) $(libccf) $(liblogccf) $(libspecfunc)

$(libconvol) : convol.cpp types.cpp
	$(CC) convol.cpp types.cpp -o $(libconvol) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11) $(GSL)

$(librebin) : rebin.cpp types.cpp
	$(CC) rebin.cpp types.cpp -o $(librebin) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11)

$(libccf) : iccf.cpp types.cpp
	$(CC) iccf.cpp types.cpp -o $(libccf) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11)

$(liblogccf) : logccf.cpp cppspecfunc.cpp types.cpp
	$(CC) logccf.cpp cppspecfunc.cpp types.cpp -o $(liblogccf) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11) $(FFTW) $(GSL)

$(libspecfunc) : cppspecfunc.cpp types.cpp
	$(CC) cppspecfunc.cpp types.cpp -o $(libspecfunc) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11) $(GSL)

clean :
	rm $(libconvol) $(librebin) $(libccf) $(liblogccf) $(libspecfunc)
