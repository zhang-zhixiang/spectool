#include <cmath>
#include <tuple>
#include <vector>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <complex>
#include <fftw3.h>
// #include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

typedef std::vector<double> VEC;
typedef const std::vector<double> CVEC;

class Shift_spec {
    // The code refer the example code in blog https://www.cnblogs.com/aiguona/p/9407425.html
private:
    Shift_spec(const Shift_spec&);
    Shift_spec& operator=(const Shift_spec&); //prevent copy
    std::complex<double> * _in, * _out;
    std::complex<double> * _fourtr, *_four_with_shift;
    fftw_plan _p;
    fftw_plan _ifft;
    size_t _size;
    double * _sig;
    double _shift;
    void ini_par(int size);
    void destroy_par();
public:
    double * specout;

    Shift_spec(CVEC & spec);
    bool set_spec(CVEC & spec);
    double get_shift_value();
    double * get_shift_spec(double shift);
    VEC get_shift_spec_arr(double shift);
    ~Shift_spec();
};

Shift_spec::Shift_spec(CVEC & spec):
    _in(nullptr), _out(nullptr), _fourtr(nullptr),
    _four_with_shift(nullptr), 
    _p(nullptr), _ifft(nullptr),
    _size(0), _sig(nullptr), _shift(0){
    set_spec(spec);
}

void Shift_spec::destroy_par(){
    if (_in != nullptr){
        fftw_destroy_plan(_p);
        fftw_destroy_plan(_ifft);
        delete [] _in;
        delete [] _out;
        delete [] _fourtr;
        delete [] _four_with_shift;
        delete [] _sig;
        delete [] specout;
        _p = nullptr;
        _ifft = nullptr;
        _in = nullptr;
        _out = nullptr;
        _fourtr = nullptr;
        _four_with_shift = nullptr;
        _sig = nullptr;
        specout = nullptr;
        _size = 0;
    }
}

Shift_spec::~Shift_spec(){
    destroy_par();
}

void Shift_spec::ini_par(int size){
    if (_in != nullptr) destroy_par();
    _size = size;
    // _in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    // _out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    _in = new std::complex<double>[_size];
    _out = new std::complex<double>[_size];
    _fourtr = new std::complex<double>[_size];
    _four_with_shift = new std::complex<double>[_size];
    _ifft = fftw_plan_dft_1d(_size,
                             reinterpret_cast<fftw_complex*>(_in),
                             reinterpret_cast<fftw_complex*>(_fourtr),
                             FFTW_BACKWARD, FFTW_ESTIMATE);
    _p = fftw_plan_dft_1d(_size,
                          reinterpret_cast<fftw_complex*>(_four_with_shift),
                          reinterpret_cast<fftw_complex*>(_out),
                          FFTW_FORWARD, FFTW_ESTIMATE);
    _sig = new double[_size];
    specout = new double[_size];

    double doublesize = double(_size);
    int half_length = _size / 2;
    for(size_t ind = 0; ind < _size; ++ind){
        auto ins = (ind + half_length) % _size;
        _sig[ins] = (ind / doublesize - 0.5) * 2 * M_PI;
    }
}

bool Shift_spec::set_spec(CVEC & spec){
    if (_in == nullptr || _size != spec.size())
        ini_par(spec.size());
    for(size_t ind = 0; ind < _size; ++ind){
        reinterpret_cast<double*>(_in)[2*ind] = spec[ind];
        reinterpret_cast<double*>(_in)[2*ind+1] = 0;
    }
    fftw_execute(_ifft);
    return true;
}

double * Shift_spec::get_shift_spec(double shift){
    _shift = shift;
    for (size_t ind = 0; ind < _size; ++ind){
        auto a = cos(_sig[ind]*shift);
        auto b = sin(_sig[ind]*shift);
        std::complex<double> complexshift = {a, b};
        _four_with_shift[ind] = _fourtr[ind] * complexshift;
    }
    fftw_execute(_p);
    for(size_t ind = 0; ind < _size; ++ind)
        specout[ind] = std::abs(_out[ind]) / _size;
    return specout;
}

VEC Shift_spec::get_shift_spec_arr(double shift){
    get_shift_spec(shift);
    VEC out(_size);
    for(size_t ind = 0; ind < _size; ++ind)
        out[ind] = specout[ind];
    return out;
}

template < typename Iter, typename Iterb>
auto get_ccf_value(Iter begin, Iter end, Iterb begin_ref, bool mult=true){
    typename std::iterator_traits<Iter>::value_type out = 0;
    if(mult == true){
        for(size_t ind = 0; begin+ind != end; ++ind)
            out += *(begin+ind) * (*(begin_ref+ind));
    } else {
        for(size_t ind = 0; begin+ind != end; ++ind){
            auto dif = *(begin+ind) - *(begin_ref+ind);
            out += dif*dif;
        }
    }
    return out;
}

auto get_shift(CVEC & spec, CVEC & spec_ref, double left_edge, double right_edge, double resolution, bool mult=true){
    // return shift_peak, rmax
    int lefte = int(std::floor(left_edge));
    int righte = int(std::ceil(right_edge));
    int range = std::max(std::abs(lefte), std::abs(righte));
    int lenshift = righte - lefte + 1;
    std::vector<int> shiftlst(lenshift);
    for(int ind = 0; ind < lenshift; ++ind) shiftlst[ind] = lefte + ind;
    std::vector<double> rlst;
    for (int shift = lefte; shift <= righte; ++shift){
        auto sfrom = spec.begin() + range;
        auto send = spec.end() - range;
        auto tfrom = spec_ref.begin() + range + shift;
        auto r = get_ccf_value(sfrom, send, tfrom, mult);
        rlst.push_back(r);
    }
    int indminmax = 0;
    if (mult == true){
        auto itrmax = std::max_element(rlst.begin(), rlst.end());
        indminmax = std::distance(rlst.begin(), itrmax);
    } else {
        auto itrmin = std::min_element(rlst.begin(), rlst.end());
        indminmax = std::distance(rlst.begin(), itrmin);
    }
    int aprox_shift = lefte + indminmax;
    if (resolution > 1)
        return std::make_tuple(double(aprox_shift), rlst[indminmax]);
    VEC rlst2, shiftlst2;
    Shift_spec shiftmodel(spec_ref);
    for(double shift = aprox_shift-1; shift < aprox_shift+1; shift+=resolution){
        shiftlst2.push_back(shift);
        shiftmodel.get_shift_spec(shift);
        auto sfrom = spec.begin() + range;
        auto send = spec.end() - range;
        auto tbegin = shiftmodel.specout + range;
        auto r = get_ccf_value(sfrom, send, tbegin, mult);
        rlst2.push_back(r);
    }
    if (mult == true){
        auto itrmax = std::max_element(rlst2.begin(), rlst2.end());
        indminmax = std::distance(rlst2.begin(), itrmax);
    } else {
        auto itrmin = std::min_element(rlst2.begin(), rlst2.end());
        indminmax = std::distance(rlst2.begin(), itrmin);
    }
    auto finalshift = shiftlst2[indminmax];
    auto rmax = rlst2[indminmax];
    return std::make_tuple(finalshift, rmax);
}


PYBIND11_MODULE(liblogccf, m) {
    m.doc() = "Simple test";

    py::class_<Shift_spec>(m, "Shift_spec")
        .def(py::init<CVEC>())
        .def("get_shift_spec_arr", &Shift_spec::get_shift_spec_arr);

    m.def("get_shift", &get_shift, "run FR/RSS to estimate the error bar");
}