#include <cmath>
#include <tuple>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <complex>
#include <fftw3.h>

typedef std::vector<double> VEC;
typedef const std::vector<double> CVEC;

class Shift_spec {
    // The code refer the example code in blog https://www.cnblogs.com/aiguona/p/9407425.html
    std::complex<double> * _in, * _out;
    fftw_plan _p;
    size_t _size;
    double * _sig;
    double * _sh;
    std::complex<double> * _complexarr;
    double _shift;
    void ini_par(int size);
    void destroy_par();
public:
    double * specout;
    double get_shift_value();
    Shift_spec(CVEC & spec);
    bool set_spec(CVEC & spec);
    double * shift_spec(double shift);
    ~Shift_spec();
};

void Shift_spec::destroy_par(){
    if (_in != nullptr){
        fftw_destroy_plan(_p);
        // fftw_free(_in);
        // fftw_free(_out);
        delete [] _in;
        delete [] _out;
        delete [] _sig;
        delete [] _sh;
        delete [] _complexarr;
        delete [] specout;
        _p = nullptr;
        _in = nullptr;
        _out = nullptr;
        _sig = nullptr;
        _sh = nullptr;
        specout = nullptr;
        _size = 0;
    }
}

void Shift_spec::ini_par(int size){
    if (_in != nullptr) destroy_par();
    _size = size;
    // _in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    // _out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    _in = new std::complex<double>[_size];
    _out = new std::complex<double>[_size];
    _p = fftw_plan_dft_1d(_size,
                          reinterpret_cast<fftw_complex*>(_in),
                          reinterpret_cast<fftw_complex*>(_out),
                          FFTW_FORWARD, FFTW_ESTIMATE);
    _sig = new double[_size];
    _sh = new double[_size];
    _complexarr = new std::complex<double>[_size];
    specout = new double[_size];

    double doublesize = double(_size);
    int half_length = _size / 2;
    for(size_t ind = 0; ind < _size; ++ind){
        auto ins = (ind + half_length) % _size;
        _sig[ins] = (ind / doublesize - 0.5) * 2 * M_PI;
    }
}

bool Shift_spec::set_spec(CVEC & spec){
    if (_in != nullptr && _size != spec.size())
        ini_par(spec.size());
    return true;
}

double * Shift_spec::shift_spec(double shift){
    _shift = shift;
    for (size_t ind = 0; ind < _size; ++ind){
        auto a = cos(_sig[ind]*shift);
        auto b = sin(_sig[ind]*shift);
        reinterpret_cast<double*>(_complexarr)[2*ind] = a;
        reinterpret_cast<double*>(_complexarr)[2*ind + 1] = b;
    }
    for(size_t ind = 0; ind < _size; ++ind) _sh[ind] = _sig[ind] * _shift;

    return specout;
}

template < typename Iter, typename Iterb>
auto get_ccf(Iter begin, Iter end, Iterb begin_ref, bool mult=true){
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

// int main(){
//     VEC a{1,2,3,4};
//     VEC b{1,2,3,4};
//     // auto data = std::accumulate(a.begin(), a.end())
//     auto out = get_ccf(a.begin(), a.end(), b.begin());
//     return 0;
// }


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
        auto r = get_ccf(sfrom, send, tfrom, mult);
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
    for(double shift = aprox_shift-1; shift < aprox_shift+1; shift+=resolution){
        shiftlst2.push_back(shift);
        auto sfrom = spec.begin() + range;
        auto send = spec.end() - range;
        auto tbegin = spec_ref.begin();
        auto r = get_ccf(sfrom, send, tbegin, mult);
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