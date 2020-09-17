#ifndef __CPPSPECFUNC_H__
#define __CPPSPECFUNC_H__

#include <memory>
#include "types.h"
#include "gsl/gsl_multifit.h"

VEC normalize_wave(CVEC & wave);
VEC get_normalized_spec(CVEC & wave, CVEC & flux, int medsize, int order);

class Continuum{
    Continuum(const Continuum &);
    Continuum& operator=(const Continuum&);
    gsl_multifit_robust_workspace * work;
    gsl_matrix * X, * cov_par;
    gsl_vector * x, * y, * par;
    VEC _wave;
    VEC _flux;
    VEC _norm_wave;
    int _order;
    int _med_size;
    int _max_iter;
    void set_all_iter_zero();
    void ini_gsl(int specsize, int order);
    void remove_gsl();

public:
    Continuum(CVEC & wave, CVEC & flux, int order, int medsize, int max_iter);
    Continuum(CVEC & flux, int order, int medsize, int max_iter);
    Continuum(int fluxsize, int order, int medsize, int max_iter);
    ~Continuum();
    int size() const;
    bool set_spec(CVEC & wave, CVEC & flux);
    bool set_spec(CVEC & flux);
    bool set_spec(double * spec_begin, double * spec_end);
    bool set_order(int order);
    bool set_median_filter_size(int medsize);
    bool set_max_iteration(int max_iter);
    VEC get_continuum();
    VEC get_norm_spec();
    py::array_t<double> get_continuum_arr();
    py::array_t<double> get_norm_spec_arr();
    std::shared_ptr<double[]> get_continuum_itr();
    std::shared_ptr<double[]> get_norm_spec_itr();
    std::shared_ptr<double[]> specout_itr;
    std::shared_ptr<double[]> contout_itr;
};


#endif // !__CPPSPECFUNC_H__