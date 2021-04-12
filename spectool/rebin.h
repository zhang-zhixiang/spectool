#ifndef __REBIN_H__
#define __REBIN_H__

#include <numeric>
#include <vector>

#define DARR std::vector<double>

DARR rebin(const DARR& wave, const DARR& flux, const DARR& new_wave);
DARR rebin_err(const DARR& wave, const DARR& err, const DARR& new_wave);
DARR rebin_padvalue(const DARR& wave, const DARR& flux, const DARR& new_wave);

#endif  // !__REBIN_H