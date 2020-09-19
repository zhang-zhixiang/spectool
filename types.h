#ifndef __SPECTOOL_TYPES__H__
#define __SPECTOOL_TYPES__H__

#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

typedef std::vector<double> VEC;
typedef const std::vector<double> CVEC;

namespace py = pybind11;

py::array_t<double> VEC2numpyarr(const std::vector<double> & vec);

std::vector<double> numpyarr2VEC(const py::array_t<double> & input);

template <typename Iter>
inline py::array_t<double> VEC2numpyarr(Iter begin, Iter end){
    int size = end - begin;
    auto result = py::array_t<double>(size);
    py::buffer_info buf = result.request();
    double *ptr = (double *) buf.ptr;
    for (size_t ind = 0; ind < size; ++ind)
        ptr[ind] = *(begin+ind);
    return result;
}

#endif