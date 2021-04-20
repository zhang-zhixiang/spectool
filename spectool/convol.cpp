#include "convol.h"
#include <gsl/gsl_sf_legendre.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>
#include <tuple>
#include <deque>
#include <vector>
#include "pybind11/stl.h"
#include "types.h"
#include "rebin.h"

// Please ensure -1 < arrx < 1
VEC legendre_poly(CVEC& arrx, CVEC& arrpar) {
  VEC arrresult(arrx.size());
  for (auto& val : arrresult)
    val = 0;
  double* buff = new double[arrpar.size()];
  double order = arrpar.size() - 1;
  for (size_t ind = 0; ind < arrx.size(); ++ind) {
    gsl_sf_legendre_Pl_array(order, arrx[ind], buff);
    double temp = 0;
    for (size_t nn = 0; nn < arrpar.size(); ++nn)
      temp += arrpar[nn] * buff[nn];
    arrresult[ind] = temp;
  }
  delete[] buff;
  return arrresult;
}

VEC poly(CVEC& arrx, CVEC& arrpar) {
  VEC arrret(arrx.size());
  for (auto& val : arrret)
    val = 0;
  for (size_t ind = 0; ind < arrpar.size(); ++ind) {
    double par = arrpar[ind];
    if (par != 0)
      for (size_t j = 0; j < arrret.size(); ++j) {
        double comp = par * pow(arrx[j], ind);
        arrret[j] += comp;
      }
  }
  return arrret;
}

// Dlambda = a0 + a1*lam + a2*lam^2 + a3*lam3 ...
// return lambda + Dlambda
VEC map_wave(CVEC& wave, CVEC& map_par) {
  VEC new_wave = poly(wave, map_par);
  for (size_t ind = 0; ind < new_wave.size(); ++ind)
    new_wave[ind] += wave[ind];
  return new_wave;
}

inline double gaussian(double x, double sigma, double x0) {
  // return 1/sqrt(2*M_PI)/sigma * exp(-(x-x0)*(x-x0)/(2*sigma*sigma));
  return exp(-(x - x0) * (x - x0) / (2 * sigma * sigma));
}

VEC gaussian(CVEC& arrx, double sigma, double x0) {
  VEC arrret(arrx.size());
  for (size_t ind = 0; ind < arrx.size(); ++ind) {
    double x = arrx[ind];
    arrret[ind] = gaussian(x, sigma, x0);
  }
  return arrret;
}

std::pair<int, int> gaussian2(CVEC& arrx,
                              double sigma,
                              double x0,
                              VEC& result,
                              size_t index_ref,
                              double threshold_ratio) {
  // double x_ref = arrx[index_ref];
  double val_ref = gaussian(x0, sigma, x0);
  double valtmp = val_ref;
  double threshold = val_ref * threshold_ratio;
  int indl = index_ref;
  result[indl] = valtmp;
  while (valtmp > threshold) {
    --indl;
    if (indl < 0) {
      indl = 0;
      break;
    }
    valtmp = gaussian(arrx[indl], sigma, x0);
    result[indl] = valtmp;
  };
  int indr = index_ref + 1;
  valtmp = val_ref;
  while (indr < arrx.size() && valtmp > threshold) {
    valtmp = gaussian(arrx[indr], sigma, x0);
    result[indr++] = valtmp;
  }
  return std::make_pair(indl, indr);
}

VEC _get_edge(CVEC& wave) {
  VEC interval(wave.size());
  std::adjacent_difference(wave.begin(), wave.end(), interval.begin());
  interval[0] = interval[1];
  for (auto& val : interval)
    val *= 0.5;
  VEC edge_out(wave.size() + 1);
  for (size_t ind = 0; ind < wave.size(); ++ind)
    edge_out[ind] = wave[ind] - interval[ind];
  edge_out.back() = wave.back() + interval.back();
  return edge_out;
}

VEC gauss_filter_mutable(CVEC& wave,
                         CVEC& flux,
                         CVEC& arrvelocity) {
  const double c = 299792458.0 / 1000;
  const double sigma_l = arrvelocity.front() / 2.355 * wave.front() / c;
  const double sigma_r = arrvelocity.back() / 2.355 * wave.back() / c;
  double delta_w1 = *(wave.begin() + 1) - *(wave.begin());
  double delta_w2 = *(wave.rbegin()) - *(wave.rbegin() + 1);
  int left_margin = (int)(5 * sigma_l / delta_w1);
  int right_margin = (int)(5 * sigma_r / delta_w2);
  VEC newave, newflux, arrsigma;
  double wtmp = wave.front() - left_margin * delta_w1;
  for (int i = 0; i < left_margin; ++i) {
    newave.push_back(wtmp);
    newflux.push_back(flux.front());
    double sigma = arrvelocity.front() / 2.355 * wtmp / c;
    arrsigma.push_back(sigma);
    wtmp += delta_w1;
  }
  for (int i = 0; i < wave.size(); ++i) {
    newave.push_back(wave[i]);
    newflux.push_back(flux[i]);
    double sigma = arrvelocity[i] / 2.355 * wave[i] / c;
    arrsigma.push_back(sigma);
  }
  wtmp = wave.back();
  for (int i = 0; i < right_margin; ++i) {
    wtmp += delta_w2;
    newave.push_back(wtmp);
    newflux.push_back(flux.back());
    double sigma = arrvelocity.back() / 2.355 * wtmp / c;
    arrsigma.push_back(sigma);
  }
  VEC gauss_profile(newave.size());
  VEC new_flux(newave.size());
  for (auto& val : new_flux)
    val = 0;
  VEC arredge = _get_edge(newave);
  VEC arrwidth;
  for (size_t d = 0; d < arredge.size() - 1; ++d)
    arrwidth.push_back(arredge[d + 1] - arredge[d]);
  for (size_t ind = 0; ind < newave.size(); ++ind) {
    double sigma = arrsigma[ind];
    double w0 = newave[ind];
    double mf = newflux[ind] * arrwidth[ind];
    auto indlr = gaussian2(newave, sigma, w0, gauss_profile, ind, 1.0e-5);
    const int indl = indlr.first;
    const int indr = indlr.second;
    double area = 0;
    for (size_t j = indl; j < indr; ++j)
      area += arrwidth[j] * gauss_profile[j];
    double inv_area = 1.0 / area;
    for (size_t j = indl; j < indr; ++j) {
      new_flux[j] += mf * (gauss_profile[j] * inv_area);
      gauss_profile[j] = 0;
    }
  }
  VEC outflux;
  for (int i = left_margin; i < left_margin + wave.size(); ++i)
    outflux.push_back(new_flux[i]);
  return outflux;
}

VEC _get_width(CVEC& wave){
  VEC out(wave.size());
  for(int ind = 1; ind < wave.size() - 1; ++ind){
    out[ind] = (wave[ind+1] - wave[ind-1]) / 2.0;
  }
  out[0] = wave[1] - wave[0];
  const int last = wave.size() - 1;
  out[last] = wave[last] - wave[last-1];
  return out;
}

void inline to_zero(VEC & arr){
  for(auto & val : arr)
    val = 0;
}

void inline vel_to_lambda(CVEC & velocity, const double w, VEC & lambda_arr){
  for(int ind = 0; ind < velocity.size(); ++ind){
    double myvel = velocity[ind];
    double delta_w = myvel / C_VAC * w;
    lambda_arr[ind] = w + delta_w;
  }
}

template<typename IT, typename IT2>
double inline _sum_flux(IT iflux, IT iflux_end, IT2 iwidth){
  double out = 0;
  while (iflux != iflux_end)
  {
    out += *(iflux++) * (*(iwidth++));
  }
  return out;
}

auto add_margin(CVEC& wave, CVEC& flux, CVEC& velocity){
  const double delta_w = (wave.back() - wave.front()) / wave.size();
  const double vl = velocity.front();
  const double vr = velocity.back();
  const double delta_wl = 3 * vl / C_VAC * wave.front();
  const double delta_wr = 3 * vr / C_VAC * wave.back();
  double wl = wave.front() + delta_wl;
  double wr = wave.back() + delta_wr;
  if(wl < 0){
    double siz = std::ceil(-wl/delta_w) + 1;
    wl += siz * delta_w;
  }
  std::deque<double> new_wave(wave.begin(), wave.end());
  std::deque<double> new_flux(flux.begin(), flux.end());
  int left_margin = 0;
  while(new_wave.front() > wl){
    left_margin++;
    new_wave.push_front(new_wave.front() - delta_w);
    new_flux.push_front(new_flux.front());
  }
  while(new_wave.back() < wr){
    new_wave.push_back(new_wave.back() + delta_w);
    new_flux.push_back(new_flux.back());
  }
  VEC out_wave(new_wave.begin(), new_wave.end());
  VEC out_flux(new_flux.begin(), new_flux.end());
  return std::make_tuple(out_wave, out_flux, left_margin);
}

VEC filter_use_given_profile(CVEC& wave, CVEC& flux, CVEC& velocity, CVEC& profile){
  auto [new_wave, new_flux, left_margin] = add_margin(wave, flux, velocity);
  VEC out(new_flux.size());
  VEC lambda_arr(velocity.size());
  CVEC width_VEC = _get_width(new_wave);
  to_zero(out);
  for(int ind=0; ind< new_wave.size(); ++ind){
    double w = new_wave[ind];
    vel_to_lambda(velocity, w, lambda_arr);
    VEC outprofile = rebin(lambda_arr, profile, new_wave);
    double sumflux = _sum_flux(outprofile.begin(), outprofile.end(), width_VEC.begin());
    if (sumflux == 0){
      out[ind] = new_flux[ind];
    } else {
      double intflux = width_VEC[ind] * new_flux[ind];
      for(auto & val : outprofile) val *= intflux/sumflux;
      for(int j = 0; j < out.size(); ++j) out[j] += outprofile[j];
    }
  }
  auto from = out.begin() + left_margin;
  auto end = from + wave.size();
  VEC new_out(from, end);
  return new_out;
}

// a gaussian filter for spectrum, the sigma of gaussians can be different in
// different wave, the sigma is defined as sigma = par0 + par1*wave +
// par2*wave^2 ... return the fluxes after smooth
VEC gauss_filter(CVEC& wave, CVEC& flux, CVEC& arrpar) {
  VEC arrsigma = poly(wave, arrpar);
  // adjust for boundry condition
  auto left_sigma = arrsigma.front();
  auto right_sigma = arrsigma.back();
  double delta_w1 = *(wave.begin() + 1) - *(wave.begin());
  double delta_w2 = *(wave.rbegin()) - *(wave.rbegin() + 1);
  int left_margin = (int)(5 * left_sigma / delta_w1);
  int right_margin = (int)(5 * right_sigma / delta_w2);
  VEC newave, newflux;
  double wtmp = wave.front() - left_margin * delta_w1;
  for (int i = 0; i < left_margin; ++i) {
    newave.push_back(wtmp);
    newflux.push_back(flux.front());
    wtmp += delta_w1;
  }
  for (int i = 0; i < wave.size(); ++i) {
    newave.push_back(wave[i]);
    newflux.push_back(flux[i]);
  }
  wtmp = wave.back();
  for (int i = 0; i < right_margin; ++i) {
    wtmp += delta_w2;
    newave.push_back(wtmp);
    newflux.push_back(flux.back());
  }

  arrsigma = poly(newave, arrpar);
  VEC gauss_profile(newave.size());
  VEC new_flux(newave.size());
  for (auto& val : new_flux)
    val = 0;
  VEC arredge = _get_edge(newave);
  VEC arrwidth;
  for (size_t d = 0; d < arredge.size() - 1; ++d)
    arrwidth.push_back(arredge[d + 1] - arredge[d]);
  for (size_t ind = 0; ind < newave.size(); ++ind) {
    double sigma = arrsigma[ind];
    double w0 = newave[ind];
    double mf = newflux[ind] * arrwidth[ind];
    auto indlr = gaussian2(newave, sigma, w0, gauss_profile, ind, 1.0e-5);
    const int indl = indlr.first;
    const int indr = indlr.second;
    double area = 0;
    for (size_t j = indl; j < indr; ++j)
      area += arrwidth[j] * gauss_profile[j];
    double inv_area = 1.0 / area;
    for (size_t j = indl; j < indr; ++j) {
      new_flux[j] += mf * (gauss_profile[j] * inv_area);
      gauss_profile[j] = 0;
    }
  }
  VEC outflux;
  for (int i = left_margin; i < left_margin + wave.size(); ++i)
    outflux.push_back(new_flux[i]);
  return outflux;
}

py::array_t<double> numpy_gauss_filter(CVEC& wave,
                                       CVEC& flux,
                                       CVEC& arrpar) {
  return VEC2numpyarr(gauss_filter(wave, flux, arrpar));
}

py::array_t<double> numpy_gauss_filter_mutable(CVEC& wave,
                                               CVEC& flux,
                                               CVEC& arrvelocity) {
  return VEC2numpyarr(gauss_filter_mutable(wave, flux, arrvelocity));
}

py::array_t<double> numpy_legendre_poly(CVEC& arrx, CVEC& arrpar) {
  return VEC2numpyarr(legendre_poly(arrx, arrpar));
}

py::array_t<double> numpy_filter_use_given_profile(CVEC& wave,
                                                   CVEC& flux,
                                                   CVEC& velocity,
                                                   CVEC& profile){
  return VEC2numpyarr(filter_use_given_profile(wave, flux, velocity, profile));
}

PYBIND11_MODULE(convol, m) {
  // xt::import_numpy();
  m.doc() = " ";

  m.def("gauss_filter", numpy_gauss_filter,
        "Smooth the input spectrum\ninput par: wave, flux, arrpar");
  m.def("gauss_filter_mutable", numpy_gauss_filter_mutable,
        "Smooth the input spectrum\ninput par: wave, flux, arrvelocity");
  m.def("legendre_poly", numpy_legendre_poly,
        "legendre polynomial function\ninput par: arrx, arrpar\nCaution: "
        "Please ensure -1 < arrx < 1!");
  m.def("filter_use_given_profile", numpy_filter_use_given_profile,
        "Smooth the input spectrum using the given kernel\n"
        "input par\n"
        "wave(numpy.ndarray): the wavelength of the spectrum\n"
        "flux(numpy.ndarray): the flux of spectrum\n"
        "velocity(numpy.ndarray): the velocity of the convolution kernel(in the unit of km/s)\n"
        "profile(numpy.darray): the profile of the convolution kernel"
        );
}