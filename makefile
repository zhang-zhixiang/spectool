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
GSL := $(shell pkg-config --libs gsl)
SHARE = -shared

librebin = rebin$(suffix)
libconvol = convol$(suffix)
libccf = libccf$(suffix)

default : $(libconvol) $(librebin) $(libccf)

$(libconvol) : convol.cpp
	$(CC) convol.cpp -o $(libconvol) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11) $(GSL)

$(librebin) : rebin.cpp
	$(CC) rebin.cpp -o $(librebin) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11)

$(libccf) : iccf.cpp
	$(CC) iccf.cpp -o $(libccf) $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11)

clean :
	rm $(libconvol) $(librebin) $(libccf)
