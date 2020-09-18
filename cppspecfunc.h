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
    gsl_vector * cache;
    // VEC _wave;
    VEC _flux;
    VEC _norm_wave;
    int _order;
    int _med_size;
    int _max_iter;
    void set_all_iter_zero();
    void ini_gsl(size_t specsize, size_t order);
    void remove_gsl();
    void fill_X();

public:
    Continuum(CVEC & wave, CVEC & flux, size_t order, size_t medsize, size_t max_iter);
    Continuum(CVEC & flux, size_t order, size_t medsize, size_t max_iter);
    Continuum(int fluxsize, size_t order, size_t medsize, size_t max_iter);
    ~Continuum();
    int size() const;
    void set_spec(CVEC & wave, CVEC & flux);
    void set_spec(CVEC & flux);
    void set_spec(double * spec_begin, double * spec_end);
    void set_order(size_t order);
    void set_median_filter_size(size_t medsize);
    void set_max_iteration(size_t max_iter);
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