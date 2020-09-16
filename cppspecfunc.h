#ifndef __CPPSPECFUNC_H__
#define __CPPSPECFUNC_H__

#include <memory>
#include "types.h"

VEC normalize_wave(CVEC & wave);

class Continuum{
    VEC _wave;
    VEC _flux;

public:
    Continuum(CVEC & wave, CVEC & flux);
    ~Continuum();
    int get_size() const;
    bool set_spec(CVEC & wave, CVEC & flux);
    bool set_order(int order);
    bool set_max_iteration(int max_iter);
    VEC get_continuum();
    VEC get_norm_spec();
    py::array_t<double> get_continuum_arr();
    py::array_t<double> get_norm_spec_arr();
}


#endif // !__CPPSPECFUNC_H__