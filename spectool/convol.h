#ifndef __CONVOL_H__
#define __CONVOL_H__

#include <vector>
#include "types.h"

VEC poly(CVEC& arrx, CVEC& arrpar);

VEC map_wave(CVEC& wave, CVEC& map_par);

VEC gauss_filter(CVEC& wave, CVEC& flux, CVEC& arrpar);

VEC gauss_filter_mutable(CVEC& wave,
                         CVEC& flux,
                         CVEC& arrvelocity);

VEC filter_use_given_profile(CVEC& wave,
                             CVEC& flux,
                             CVEC& velocity,
                             CVEC& profile);

VEC legendre_poly(CVEC& arrx, CVEC& arrpar);

#endif
