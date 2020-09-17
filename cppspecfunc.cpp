#include "cppspecfunc.h"


VEC normalize_wave(CVEC & wave){
    double med = (wave.front() + wave.back()) / 2;
    double length = wave.back() - wave.front();
    VEC out(wave.size());
    for(size_t ind = 0; ind < wave.size(); ++ind)
        out[ind] = (wave[ind] - med) / length * 1.9999999;
    return out;
}

Continuum::Continuum(CVEC & wave, CVEC & flux){
    _wave = wave;
}