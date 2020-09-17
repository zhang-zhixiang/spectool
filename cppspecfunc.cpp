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
    cache = nullptr;
    par = nullptr;
    specout_itr = nullptr;
    contout_itr = nullptr;
}

void Continuum::ini_gsl (size_t specsize, size_t order){
    remove_gsl();
    work = gsl_multifit_robust_alloc(
            gsl_multifit_robust_bisquare, specsize, order+1);
    X = gsl_matrix_alloc(specsize, order+1);
    cov_par = gsl_matrix_alloc(order+1, order+1);
    par = gsl_vector_alloc(order+1);
    cache = gsl_vector_alloc(order+1);
    specout_itr.reset(new double[specsize]);
    contout_itr.reset(new double[specsize]);
}

void Continuum::remove_gsl(){
    if ( work != nullptr){
        gsl_multifit_robust_free(work);
        gsl_matrix_free(X);
        gsl_matrix_free(cov_par);
        gsl_vector_free(par);
        gsl_vector_free(cache);
    }
    set_all_iter_zero();
}

void Continuum::fill_X(){
    for (size_t ind = 0; ind < _norm_wave.size(); ++ind){
        double wa = _norm_wave[ind];
        gsl_sf_legendre_Pl_array(_order, wa, cache->data);
        gsl_matrix_set_row(X, ind, cache);
    }
}

Continuum::Continuum(CVEC & wave, CVEC & flux, size_t order, size_t medsize, size_t max_iter){
    _norm_wave = normalize_wave(wave);
    _flux = flux;
    _order = order;
    _med_size = medsize;
    _max_iter = max_iter;
    set_all_iter_zero();
    ini_gsl(_norm_wave.size(), _order);
    fill_X();
}

VEC get_norm_wave(size_t specsize){
    double step = 1.9999999 / specsize;
    VEC out;
    for (size_t ind = 0; ind < specsize; ++ind)
        out.push_back(-0.99999999999 + ind * step);
    return out;
}

Continuum::Continuum(CVEC & flux, size_t order, size_t medsize, size_t max_iter){
    _norm_wave = get_norm_wave(flux.size());
    _flux = flux;
    _order = order;
    _med_size = medsize;
    _max_iter = max_iter;
    set_all_iter_zero();
    ini_gsl(_norm_wave.size(), _order);
    fill_X();
}


Continuum::Continuum(int fluxsize, size_t order, size_t medsize, size_t max_iter){
    _norm_wave = get_norm_wave(fluxsize);
    _order = order;
    _med_size = medsize;
    _max_iter = max_iter;
    set_all_iter_zero();
    ini_gsl(_norm_wave.size(), _order);
    fill_X();
}

Continuum::~Continuum(){
    remove_gsl();
}

int Continuum::size() const {
    return _norm_wave.size();
}

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