#include <math.h>
#include <stdio.h>
#include <vector>
// #include <matplot/matplot.h>
// #include <xtensor/xarray.hpp>
// #include <xtensor/xcsv.hpp>
// #include <xtensor/xadapt.hpp>
// #include <xtensor/xview.hpp>
// #include <istream>
// #include <fstream>
// #include <iostream>
#include "pybind11/stl.h"
#include "types.h"

const double c = 299792.458; // light velocity in the unit of km/s
typedef std::vector<double> ARR;

int get_rotation_profile_size(double rotation, double dll)
/**
 * @brief Get the size of the rotation profile
 * @param rotation: rotation velocity in the unit of km/s
 * @param dll: delta ln lambda, delta vel = c * dll
 * @retval the size of the rotation profile from left edge to right edge
 */
{
    double dl = rotation / c;
    int ind_center = trunc(dl / dll);
    return 2 * ind_center + 1;
}

double inline G(double x, double limb)
/**
 * @brief The rotation profile kernel
 * @param x: x = (i - ind_center) * dll * c / rotation (from -1 to 1)
 * @param limb: limb darkening coefficient
 * @retval G(x)
 */
{
    double val1 = 2 * (1 - limb) * sqrt(1 - x * x) + M_PI * limb / 2 * (1 - x * x);
    double val2 = M_PI * (1 - limb/3);
    return val1 / val2;
}

int make_rotation_profile(double rotation, double limb, double dll, double * profile)
/**
 * @brief Make rotation profile
 * @param rotation: rotation velocity in the unit of km/s
 * @param limb: limb darkening coefficient
 * @param dll: delta ln lambda, delta vel = c * dll
 * @param profile: used to store the rotation profile, 
 *        the length of the array should be > 2 * ind_center + 1,
 *        where ind_center is the index of the center of the profile, 
 *        and ind_center = int(rotation / c / dll)
 * @note  The rotation profile is calculated, see equation 3 in the paper
 *        https://ui.adsabs.harvard.edu/abs/2011A%26A...531A.143D/abstract
 * @retval the index of the center of the profile, ind_center
 */
{
    double dl = rotation / c;
    int ind_center = trunc(dl / dll);
    double sum = 0;
    for (int i = 0; i <= ind_center; i++)
    {
        double x = (i - ind_center) * dll * c / rotation;
        profile[i] = G(x, limb);
        sum += profile[i];
    }
    for (int i = ind_center + 1; i < 2 * ind_center + 1; i++)
    {
        profile[i] = profile[2 * ind_center - i];
        sum += profile[i];
    }
    // printf("sum = %f\n", sum);
    for (int i = 0; i < 2 * ind_center + 1; i++)
    {
        profile[i] /= sum;
    }
    return ind_center;
}

ARR prolongate(const ARR &flux, int size)
/**
 * @brief Prolongate the both boundaries of the flux array a size of size     
 * @note  The function is like the following example:
 *        flux = [1, 2, 3, 4, 5]
 *        size = 3
 *        prolongate(flux, size) = [1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5]
 * @param flux: the flux array
 * @param size: the size of the prolongation size on each side
 * @retval the prolonged flux array
 */
{
    ARR ret(size * 2 + flux.size());
    for(int i = 0; i < size; i++)
    {
        ret[i] = flux[0];
    }
    for(int i = size; i < size + flux.size(); i++)
    {
        ret[i] = flux[i - size];
    }
    for(int i = size + flux.size(); i < size * 2 + flux.size(); i++)
    {
        ret[i] = flux[flux.size() - 1];
    }
    return ret;
}

ARR rotation_filter(const ARR &flux, double dll, double rotation, double limb)
/**
 * @brief  broadening the spectrum with a rotation profile
 * @note   the function require the wavelength cadence to be uniform 
 *         in the log space
 * @param  flux: the flux array
 * @param  dll: delta ln lambda, delta vel = c * dll
 *         Be careful: dll is the sampling interval in ln, not in log10!!!
 * @param  rotation: rotation velocity in the unit of km/s
 * @param  limb: limb darkening coefficient
 * @retval the broadened flux array
 */
{
    // printf("rotation = %f, limb = %f, dll = %f\n", rotation, limb, dll);
    int profile_size = get_rotation_profile_size(rotation, dll);
    double * profile = new double[profile_size];
    int ind_center = make_rotation_profile(rotation, limb, dll, profile);
    ARR pflux = prolongate(flux, ind_center);
    ARR broad_flux(pflux.size());
    for (int i = 0; i < pflux.size(); i++)
    {
        double sum = 0;
        for (int j = 0; j < profile_size; j++)
        {
            int indp = i + j - ind_center;
            if (indp > 0 && indp < pflux.size())
            {
                sum += profile[j] * pflux[indp];
            }
        }
        broad_flux[i] = sum;
    }
    delete [] profile;
    auto from = broad_flux.begin() + ind_center;
    auto end = broad_flux.begin() + ind_center + flux.size();
    ARR ret(from, end);
    return ret;
}

py::array_t<double> numpy_rotation_filter(const ARR& flux, double dll, double rotation, double limb)
/**
 * @brief  broadening the spectrum with a rotation profile
 * @note   the function require the wavelength cadence to be uniform 
 *         in the log space
 * @param  flux: the flux array
 * @param  dll: delta ln lambda, delta vel = c * dll
 *         Be careful: dll is the sampling interval in ln, not in log10!!!
 * @param  rotation: rotation velocity in the unit of km/s
 * @param  limb: limb darkening coefficient
 * @retval the broadened flux array
 */
{
    return VEC2numpyarr(rotation_filter(flux, dll, rotation, limb));
}

PYBIND11_MODULE(lnspecfilter, m) {
    m.doc() = "broadening spectrum in log space";

    m.def("rotation_filter", &numpy_rotation_filter, 
          "Broadening the spectrum with a rotation profile.\n"
          "@note   The function require the wavelength cadence to be uniform in the log space.\n"
          "@param  flux: the flux array\n"
          "@param  dll: delta ln lambda, delta vel = c * dll.\n        Be careful: dll is the sampling interval in ln, not in log10!!!\n"
          "@param  rotation: rotation velocity in the unit of km/s\n"
          "@param  limb: limb darkening coefficient\n"
          "@retval the broadened flux array"
          );
}

// int main()
// {
//     using namespace matplot;
//     const double rotation = 500;
//     const double limb = 0.5;
//     // double limb1 = 0.2;
//     // double limb2 = 0.8;
//     // const double dll = 0.00001;
//     // ARR spec(4000);
//     // spec[1000] = 10;
//     // spec[3000] = 10;
//     // spec[3001] = 10;
//     // spec[3002] = 10;
//     // ARR nflux = rotation_filter(spec, dll, rotation, limb);
//     // ARR logwave(nflux.size());
//     // for(int i = 0; i < logwave.size(); ++i){
//     //     logwave[i] = exp(8.6 + i * dll);
//     // }
//     // auto p = plot(logwave, nflux);
//     // p->line_width(2);
//     // show();

//     std::ifstream in_file;
//     in_file.open("test/spec_logwave_rebin.csv");
//     xt::xarray<double> data = xt::load_csv<double>(in_file);
//     // xt::xarray<double> data = xt::load_csv<double>("test/spec_logwave_rebin.csv");
//     std::cout << "size of data: " << data.size() << std::endl;
//     xt::xarray<double> wave = xt::view(data, xt::all(), 0);
//     xt::xarray<double> flux = xt::view(data, xt::all(), 1);
//     std::cout << "size of wave: " << wave.size() << std::endl;
//     std::cout << "size of flux: " << flux.size() << std::endl;
//     double ndll = log(wave(1)) - log(wave(0));
//     std::cout << "ndll = " << ndll << std::endl;
//     ARR tmpflux(flux.begin(), flux.end());
//     ARR new_flux = rotation_filter(tmpflux, ndll, rotation, limb);
//     auto pp = plot(wave, flux, wave, new_flux);
//     pp[0]->line_width(1);
//     pp[1]->line_width(1);
//     show();


//     return 0;
// }