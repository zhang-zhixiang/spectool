#include <cmath>
#include <tuple>
#include <random>
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

    Shift_spec(int size);
    Shift_spec(CVEC & spec);
    bool set_spec(CVEC & spec);
    template <typename Iter> bool set_spec(Iter begin, int size){
        if (_in == nullptr || _size != size)
            ini_par(size);
        for(size_t ind = 0; ind < _size; ++ind){
            reinterpret_cast<double*>(_in)[2*ind] = *(begin+ind);
            reinterpret_cast<double*>(_in)[2*ind+1] = 0;
        }
        fftw_execute(_ifft);
        return true;
    }
    double get_shift_value();
    double * get_shift_spec(const double shift);
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

Shift_spec::Shift_spec(int size):
    _in(nullptr), _out(nullptr), _fourtr(nullptr),
    _four_with_shift(nullptr), 
    _p(nullptr), _ifft(nullptr),
    _size(0), _sig(nullptr), _shift(0){
        ini_par(size);
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

double * Shift_spec::get_shift_spec(const double shift){
    _shift = shift;
    for (size_t ind = 0; ind < _size; ++ind){
        auto a = cos(_sig[ind]*shift);
        auto b = sin(_sig[ind]*shift);
        std::complex<double> complexshift = {a, b};
        _four_with_shift[ind] = _fourtr[ind] * complexshift;
    }
    fftw_execute(_p);
    for(size_t ind = 0; ind < _size; ++ind)
        specout[ind] = std::real(_out[ind]) / _size;
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
    auto length = end - begin;
    return out / length;
}

auto get_ccf(CVEC & spec, CVEC & spec_ref, double left_edge, double right_edge, double resolution, const bool mult=true){
    // return shift_peak, rmax
    const int lefte = int(std::floor(left_edge));
    const int righte = int(std::ceil(right_edge));
    const int range = std::max(std::abs(lefte), std::abs(righte));
    // int lenshift = righte - lefte + 1;
    VEC outshift, rlst;
    const auto sfrom = spec.begin() + range;
    const auto send = spec.end() - range;

    for (int shift = lefte; shift <= righte; ++shift){
        auto tfrom = spec_ref.begin() + range - shift;
        auto r = get_ccf_value(sfrom, send, tfrom, mult);
        outshift.push_back(double(shift));
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
    double aprox_shift = outshift[indminmax];
    // std::cout << "aprox shift = " << aprox_shift << std::endl;
    // std::cout << "resolution = " << resolution << std::endl;
    // std::cout << "range = " << range << std::endl;

    if (resolution > 1)
        return std::make_tuple(outshift, rlst);

    Shift_spec shiftmodel(spec_ref);
    const auto tbegin = shiftmodel.specout + range;
    for(double shift = aprox_shift-1; shift <= aprox_shift+1; shift+=resolution){
        shiftmodel.get_shift_spec(shift);
        auto r = get_ccf_value(sfrom, send, tbegin, mult);
        outshift.push_back(shift);
        rlst.push_back(r);
    }
    return std::make_tuple(outshift, rlst);
}

auto get_shift(CVEC & spec, CVEC & spec_ref, double left_edge, double right_edge, double resolution, bool mult=true){
    auto [vecshift, vecr] = get_ccf(spec, spec_ref, left_edge, right_edge, resolution, mult);
    int indminmax = 0;
    if (mult == true){
        auto itrmax = std::max_element(vecr.begin(), vecr.end());
        indminmax = std::distance(vecr.begin(), itrmax);
    } else {
        auto itrmin = std::min_element(vecr.begin(), vecr.end());
        indminmax = std::distance(vecr.begin(), itrmin);
    }
    double shift = vecshift[indminmax];
    double rmax = vecr[indminmax];
    return std::make_tuple(shift, rmax);
}

std::random_device r;
std::default_random_engine e1(r());

auto get_shift_mc(CVEC & spec, CVEC & spec_ref, double left_edge, double right_edge, double resolution, int mcnumber, double inc_ration, bool mult=true){
    int ccfsize = int(spec.size() * inc_ration);
    Shift_spec shiftmodel(ccfsize);
    const int lefte = int(std::floor(left_edge));
    const int righte = int(std::ceil(right_edge));
    const int range = std::max(std::abs(lefte), std::abs(righte));
    int start_window = spec.size() - ccfsize - 2*range - 1;
    std::uniform_int_distribution<int> uniform_dist(0, start_window);
    VEC outbestshiftlst, outrmaxlst;
    for (size_t loop = 0; loop < mcnumber; ++loop){
        int from = uniform_dist(e1);
        VEC outshift, rlst;
        const auto sfrom = spec.begin() + range + from;
        const auto send = spec.begin() + range + from + ccfsize;

        for (int shift = lefte; shift <= righte; ++shift){
            auto tfrom = spec_ref.begin() + range + from - shift;
            auto r = get_ccf_value(sfrom, send, tfrom, mult);
            outshift.push_back(double(shift));
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
        double aprox_shift = outshift[indminmax];

        if (resolution < 1)
            shiftmodel.set_spec(spec_ref.begin() + from, ccfsize);
            const auto tbegin = shiftmodel.specout + range;
            for(double shift = aprox_shift-1; shift <= aprox_shift+1; shift+=resolution){
                shiftmodel.get_shift_spec(shift);
                auto r = get_ccf_value(sfrom, send, tbegin, mult);
                outshift.push_back(shift);
                rlst.push_back(r);
        }

        if (mult == true){
            auto itrmax = std::max_element(rlst.begin(), rlst.end());
            indminmax = std::distance(rlst.begin(), itrmax);
        } else {
            auto itrmin = std::min_element(rlst.begin(), rlst.end());
            indminmax = std::distance(rlst.begin(), itrmin);
        }
        double shift = outshift[indminmax];
        double rmax = rlst[indminmax];
        outbestshiftlst.push_back(shift);
        outrmaxlst.push_back(rmax);
    }
    return make_tuple(outbestshiftlst, outrmaxlst);
}

PYBIND11_MODULE(liblogccf, m) {
    m.doc() = "Simple test";

    py::class_<Shift_spec>(m, "Shift_spec")
        .def(py::init<CVEC>())
        .def("get_shift_spec_arr", &Shift_spec::get_shift_spec_arr);

    m.def("get_shift", &get_shift, "get the shift of spec compared with spec_ref");
    m.def("get_shift_mc", &get_shift_mc, "get the shift lst of spec compared with spec_ref");
    m.def("get_ccf", &get_ccf, "get the ccf function array of spec compared with spec_ref");
}