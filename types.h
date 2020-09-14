#ifndef __SPECTOOL_TYPES__H__
#define __SPECTOOL_TYPES__H__

#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

py::array_t<double> VEC2numpyarr(const std::vector<double> & vec){
    auto result = py::array_t<double>(vec.size());
    py::buffer_info buf = result.request();
    double *ptr = (double *) buf.ptr;
    for (size_t ind = 0; ind < vec.size(); ++ind)
        ptr[ind] = vec[ind];
    return result;
}

std::vector<double> numpyarr2VEC(const py::array_t<double> & input){
    py::buffer_info buf = input.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");
    double *ptr = (double *) buf.ptr;
    std::vector<double> out(buf.size);
    for ( size_t ind = 0; ind < buf.size; ++ind)
        out[ind] = ptr[ind];
    return out;
}

template <typename Iter>
py::array_t<double> VEC2numpyarr(Iter begin, Iter end){
    int size = end - begin;
    auto result = py::array_t<double>(size);
    py::buffer_info buf = result.request();
    double *ptr = (double *) buf.ptr;
    for (size_t ind = 0; ind < size; ++ind)
        ptr[ind] = *(begin+ind);
    return result;
}

#endif