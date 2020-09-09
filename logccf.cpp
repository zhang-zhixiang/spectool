#include <cmath>
#include <vector>
#include <fftw3.h>

typedef std::vector<double> VEC;
typedef const std::vector<double> CVEC;

class Shift_spec {
    // The code refer the example code in blog https://www.cnblogs.com/aiguona/p/9407425.html
    fftw_complex * in, * out;
    fftw_plan p;
    size_t size;
public:
    Shift_spec(CVEC & spec);
    bool set_spec(std::vector<double> spec);
    bool get_shifted_spec(double shift);
    ~Shift_spec();
};

template < typename T>
double get_ccf(T begin, T end, T begin_ref){
    return 
}


double get_shift(CVEC & spec, CVEC & spec_ref, double left_edge, double right_edge, double resolution){
    // return shift_peak, rmax
    int lefte = int(std::floor(left_edge));
    int righte = int(std::ceil(right_edge));
    int range = std::max(std::abs(lefte), std::abs(righte));
    int lenshift = righte - lefte + 1;
    std::vector<int> shiftlst(lenshift);
    for(int ind = 0; ind < lenshift; ++ind) shiftlst[ind] = lefte + ind;
    std::vector<double> rlst;
    for (int shift = lefte; shift <= righte; ++shift){
        data
    }
    for(auto shift : shiftlst){
        da
    }
    return 0.0;
}