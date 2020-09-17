#include "pybind11/stl.h"
#include "cppspecfunc.h"
#include "gsl/gsl_filter.h"
#include "gsl/gsl_sf_legendre.h"
#include "types.h"


VEC normalize_wave(CVEC & wave){
    double med = (wave.front() + wave.back()) / 2;
    double length = wave.back() - wave.front();
    VEC out(wave.size());
    for(size_t ind = 0; ind < wave.size(); ++ind)
        out[ind] = (wave[ind] - med) / length * 1.9999999;
    return out;
}

void Continuum::set_all_iter_zero(){
    work = nullptr;
    X = nullptr;
    cov_par = nullptr;
    x = nullptr;
    y = nullptr;
    par = nullptr;
    specout_itr = nullptr;
    contout_itr = nullptr;
}

void Continuum::ini_gsl (int specsize, int order){
    // for name in 
}

Continuum::Continuum(CVEC & wave, CVEC & flux, int order, int medsize, int max_iter){
    _wave = wave;
    _norm_wave = normalize_wave(wave);
    _flux = flux;
    _order = order;
    _max_iter = max_iter;
    set_all_iter_zero();
}

// Continuum::~Continuum(){
//     remove_gsl();
// }

// int Continuum::get_size(){
//     return _wave.size();
// }

int median_filter(gsl_vector * arr, int medsize, gsl_vector * out){
    gsl_filter_rmedian_workspace * rmedian_p = gsl_filter_rmedian_alloc(medsize);
    gsl_filter_rmedian(GSL_FILTER_END_PADVALUE, arr, out, rmedian_p);
    gsl_filter_rmedian_free(rmedian_p);
    return 1;
}

VEC get_normalized_spec(CVEC & wave, CVEC & flux, int medsize=35, int order=5){
    VEC norm_wave = normalize_wave(wave);
    gsl_vector * _wave = gsl_vector_alloc(wave.size());
    gsl_vector * _flux = gsl_vector_alloc(flux.size());
    for ( size_t ind = 0; ind < _flux->size; ++ind){
        // gsl_vector_set(_wave, ind, norm_wave[ind]);
        gsl_vector_set(_flux, ind, flux[ind]);
    }
    gsl_vector * spec_med = gsl_vector_alloc(_flux->size);
    median_filter(_flux, medsize, spec_med);
    gsl_vector *par = gsl_vector_alloc(order+1);
    gsl_matrix *X = gsl_matrix_alloc(norm_wave.size(), order+1);
    gsl_matrix *cov_par = gsl_matrix_alloc(order+1, order+1);
    gsl_vector *tmp = gsl_vector_alloc(order+1);
    for (size_t ind = 0; ind < norm_wave.size(); ++ind){
        double wa = norm_wave[ind];
        gsl_sf_legendre_Pl_array(order, wa, tmp->data);
        gsl_matrix_set_row(X, ind, tmp);
    }
    gsl_multifit_robust_workspace * work = 
        gsl_multifit_robust_alloc(
            gsl_multifit_robust_bisquare, X->size1, X->size2);
    int s = gsl_multifit_robust(X, spec_med, par, cov_par, work);
    gsl_vector * y = gsl_vector_alloc(wave.size());
    gsl_vector * yerr = gsl_vector_alloc(wave.size());
    for(size_t ind = 0; ind < wave.size(); ++ind){
        gsl_vector_view v = gsl_matrix_row(X, ind);
        gsl_multifit_robust_est(&v.vector, par, cov_par, y->data+ind, yerr->data+ind);
    }
    VEC out;
    for(size_t ind = 0; ind < wave.size(); ++ind){
        double val = flux[ind] / gsl_vector_get(y, ind);
        out.push_back(val);
    }
    gsl_vector_free(_flux);
    gsl_vector_free(spec_med);
    gsl_vector_free(par);
    gsl_matrix_free(X);
    gsl_matrix_free(cov_par);
    gsl_vector_free(tmp);
    gsl_multifit_robust_free(work);
    gsl_vector_free(y);
    gsl_vector_free(yerr);
    return out;
}

py::array_t<double> numpy_get_normalized_spec(CVEC & wave, CVEC & flux, int medsize, int order){
    return VEC2numpyarr(get_normalized_spec(wave, flux, medsize, order));
}

PYBIND11_MODULE(libspecfunc, m) {
    m.doc() = "Simple test";

    m.def("get_normalized_spec", &numpy_get_normalized_spec, "get the normalized spec after removing the continuum");
}