#ifndef __CPPSPECFUNC_H__
#define __CPPSPECFUNC_H__

#include <memory>
#include "types.h"
#include "gsl/gsl_multifit.h"

VEC normalize_wave(CVEC & wave);

class Continuum{
    gsl_multifit_robust_workspace * work;
    gsl_matrix * X, * cov;
    gsl_vector * x, * y, * c;
    VEC _wave;
    VEC _flux;
    int _order;
    void ini_gsl(int specsize, int order);
    void remove_gsl();

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
};


#endif // !__CPPSPECFUNC_H__