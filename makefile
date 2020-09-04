FLAG = -fPIC -std=c++17 -O3
PY_CFLAGS := $(shell python3-config --includes)
PYBIND11 := $(shell python3 -m pybind11 --includes)
GSL = -lgsl -lgslcblas
CC = g++
SHARE = -shared

default : convol.so rebin.so libccf.so

convol.so : convol.cpp
	$(CC) convol.cpp -o convol.so $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11) $(GSL)

rebin.so : rebin.cpp
	$(CC) rebin.cpp -o rebin.so $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11)

libccf.so : iccf.cpp
	$(CC) iccf.cpp -o libccf.so $(FLAG) $(SHARE) $(PY_CFLAGS) $(PYBIND11)

clean :
	rm convol.so rebin.so libccf.so
