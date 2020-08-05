#ifndef __CONVOL_H__
#define __CONVOL_H__

#include <vector>
#define ARR std::vector<double>

ARR poly(const ARR & arrx, const ARR & arrpar);
ARR map_wave(const ARR & wave, const ARR & map_par);
ARR gauss_filter(const ARR & wave, const ARR & flux, const ARR & arrpar);
ARR gauss_filter_mutable(const ARR & wave, const ARR & flux, const ARR & arrvelocity);
ARR legendre_poly( const ARR & arrx, const ARR & arrpar);

#endif
