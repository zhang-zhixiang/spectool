#include <cmath>
#include <numeric>
#include <random>
#include <tuple>
#include <iostream>
#include <vector>
#include <algorithm>
// #include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "types.h"
// #include <gsl/gsl_statistics_double.h>


typedef std::vector<double> VEC ;

const double ivc =  1/299792.458;

template<typename IT>
inline double mean(IT from, IT aflast){
  double val = 0.0;
  size_t count = 0;
  for(auto itr = from; itr != aflast; ++itr){
    val += *itr;
    ++count;
  }
  return val / count;
}

template<typename IT>
inline double stddev(IT from, IT aflast, double mean){
  double val = 0;
  size_t count = 0;
  for(auto itr = from; itr != aflast; ++itr){
    val += (*itr-mean) * (*itr-mean);
    ++count;
  }
  return sqrt(val/count);
}

enum TYPE { SPEC, LC};

VEC new_cc(const VEC& w1, const VEC& f1, const VEC& w2, const VEC& f2, const VEC& shift, TYPE type=SPEC){
    VEC new_w1(w1);
    VEC dif(shift.size());
    std::adjacent_difference(shift.begin(), shift.end(), dif.begin());
    VEC slope;
    slope.reserve(f2.size()-1);
    for(size_t ind = 1; ind < f2.size(); ++ind)
        slope.push_back((f2[ind]-f2[ind-1]) / (w2[ind]-w2[ind-1]));
    VEC result(shift.size());
    VEC int_f2(new_w1.size());
    for(size_t ind = 0; ind < shift.size(); ++ind){
        if ( type == SPEC)
            for ( size_t aa = 0; aa < new_w1.size(); ++aa)
                new_w1[aa] += dif[ind] * w1[aa] * ivc;
        else
            for ( size_t aa = 0; aa < new_w1.size(); ++aa)
                new_w1[aa] += dif[ind];
        int indleft = 0;
        while(new_w1[indleft] < w2.front()) ++indleft;
        int indright = new_w1.size();
        while(new_w1[indright-1] >= w2.back()) --indright;
        size_t tmpid = w2.size()-1;
        for(int bb = indright-1; bb >= indleft; --bb){
            while (w2[tmpid] > new_w1[bb]) --tmpid;
            double deltaw = new_w1[bb] - w2[tmpid];
            double slop = slope[tmpid];
            double basef = f2[tmpid];
            int_f2[bb] = basef + slop * deltaw;
        }
        auto arrfrom = f1.begin()+indleft;
        auto arrend = f1.begin()+indright;
        auto brrfrom = int_f2.begin()+indleft;
        auto length = indright - indleft;
        double r = std::inner_product(arrfrom, arrend, brrfrom, 0.0);
        // double r = 0;
        // while(arrfrom != arrend) r += *arrfrom++ * *brrfrom++;
        r /= length;
        result[ind] = r;
    }
    return result;
}


double centered_lag(const VEC& shift, const VEC& ccf, double threshold=0.8){
  double lag = 0.0;
  double denominator = 0;
  double maxvalue = *std::max_element(ccf.begin(), ccf.end());
  double real_thres = maxvalue * threshold;
  for(size_t ind = 0; ind < shift.size(); ++ind)
    if ( ccf[ind] > real_thres){
      lag += ccf[ind] * shift[ind];
      denominator += ccf[ind];
    }
  return lag / denominator;
}

auto iccf_pre( const VEC& w1, const VEC& f1, const VEC& w2, const VEC& f2, const VEC& shift, TYPE type=SPEC){
  double mean1 = mean(f1.begin(), f1.end());
  // double mean1 = gsl_stats_mean(f1.begin(), 1, f1.size());
  double invstd1 = 1/stddev(f1.begin(), f1.end(), mean1);
  auto mean2 = mean(f2.begin(), f2.end());
  auto invstd2 = 1.0/stddev(f2.begin(), f2.end(), mean2);
  VEC new_f1(f1);
  for(auto & val : new_f1) val = (val - mean1) * invstd1;
  VEC new_f2(f2);
  for(auto & val : new_f2) val = (val - mean2) * invstd2;
  auto result1 = new_cc(w1, new_f1, w2, new_f2, shift, type);
  VEC shift2(shift);
  for(auto & val : shift2) val = -val;
  auto result2 = new_cc(w2, new_f2, w1, new_f1, shift2, type);
  for(size_t ind = 0; ind < result1.size(); ++ind)
    result1[ind] = (result1[ind] + result2[ind]) * 0.5;
  return result1;
}

auto iccf_mc_pre(const VEC& w1, const VEC& f1, const VEC& err1,
            const VEC& w2, const VEC& f2, const VEC& err2,
            const VEC& shift, int mc_num, TYPE type=SPEC){
  std::random_device d;
  std::default_random_engine e(d());
  std::uniform_int_distribution<int> u1(0, w1.size()-1);
  std::uniform_int_distribution<int> u2(0, w2.size()-1);
  std::vector<std::normal_distribution<double>> random_err1;
  std::vector<std::normal_distribution<double>> random_err2;
  for(auto & err : err1)
    random_err1.push_back(std::normal_distribution<double>(0, err));
  for(auto & err : err2)
    random_err2.push_back(std::normal_distribution<double>(0, err));
  bool * selarr1 = new bool[w1.size()];
  bool * selarr2 = new bool[w2.size()];
  VEC sel_w1, sel_f1, sel_w2, sel_f2;
  sel_w1.reserve(w1.size());
  sel_f1.reserve(w1.size());
  sel_w2.reserve(w2.size());
  sel_f2.reserve(w2.size());
  VEC peaklst, centerlst;
  peaklst.reserve(shift.size());
  centerlst.reserve(shift.size());
  for(size_t outloop = 0; outloop < mc_num; ++outloop){
    sel_w1.clear();
    sel_f1.clear();
    sel_w2.clear();
    sel_f2.clear();
    for(size_t ind = 0; ind < w1.size(); ++ind) selarr1[ind] = false;
    for(size_t ind = 0; ind < w2.size(); ++ind) selarr2[ind] = false;
    for(size_t ind = 0; ind < w1.size(); ++ind) selarr1[u1(e)] = true;
    for(size_t ind = 0; ind < w2.size(); ++ind) selarr2[u2(e)] = true;
    for(size_t ind = 0; ind < w1.size(); ++ind)
      if ( selarr1[ind] == true){
        sel_w1.push_back(w1[ind]);
        sel_f1.push_back(f1[ind] + random_err1[ind](e));
      }
    for(size_t ind = 0; ind < w2.size(); ++ind)
      if ( selarr2[ind] == true){
        sel_w2.push_back(w2[ind]);
        sel_f2.push_back(f2[ind] + random_err2[ind](e));
      }
    auto rcoef = iccf_pre(sel_w1, sel_f1, sel_w2, sel_f2, shift, type);
    auto indpeak = std::max_element(rcoef.begin(), rcoef.end());
    size_t step = indpeak - rcoef.begin();
    double peak = shift[step];
    double centrd = centered_lag(shift, rcoef);
    peaklst.push_back(peak);
    centerlst.push_back(centrd);
  }
  delete [] selarr1;
  delete [] selarr2;
  return std::make_tuple(peaklst, centerlst);
}

auto iccf_spec( const VEC& w1, const VEC& f1, const VEC& w2, const VEC& f2, const VEC& shift){
    return iccf_pre(w1, f1, w2, f2, shift, SPEC);
}

auto iccf( const VEC& w1, const VEC& f1, const VEC& w2, const VEC& f2, const VEC& shift){
    return iccf_pre(w1, f1, w2, f2, shift, LC);
}

auto iccf_mc_spec(const VEC& w1, const VEC& f1, const VEC& err1,
            const VEC& w2, const VEC& f2, const VEC& err2,
            const VEC& shift, int mc_num){
    return iccf_mc_pre(w1, f1, err1, w2, f2, err2, shift, mc_num, SPEC);
}

auto iccf_mc(const VEC& w1, const VEC& f1, const VEC& err1,
            const VEC& w2, const VEC& f2, const VEC& err2,
            const VEC& shift, int mc_num){
    return iccf_mc_pre(w1, f1, err1, w2, f2, err2, shift, mc_num, LC);
}

py::array_t<double> numpy_iccf(const VEC& w1, const VEC& f1, const VEC& w2, const VEC& f2, const VEC& shift){
    return VEC2numpyarr(iccf(w1, f1, w2, f2, shift));
}

auto numpy_iccf_mc(const VEC& w1, const VEC& f1, const VEC& err1,
            const VEC& w2, const VEC& f2, const VEC& err2,
            const VEC& shift, int mc_num){
    auto [peaklst, centerlst] = iccf_mc(w1, f1, err1, w2, f2, err2, shift, mc_num);
    auto numpy_peak = VEC2numpyarr(peaklst);
    auto numpy_center = VEC2numpyarr(centerlst);
    return std::make_tuple(numpy_peak, numpy_center);
}

py::array_t<double> numpy_iccf_spec( const VEC& w1, const VEC& f1, const VEC& w2, const VEC& f2, const VEC& shift){
    return VEC2numpyarr(iccf_spec(w1, f1, w2, f2, shift));
}

PYBIND11_MODULE(libccf, m)
{
    // xt::import_numpy();
    m.doc() = "A ICCF package, return ";

    m.def("iccf", numpy_iccf, "Sum the sines of the input values");
    m.def("iccf_mc", numpy_iccf_mc, "run FR/RSS to estimate the error bar");
    m.def("iccf_spec", numpy_iccf_spec, "CCF function of two spectra\ninput par:\n  wave, flux, wave_ref, flux_ref, shiftlist(velocity)");
}
