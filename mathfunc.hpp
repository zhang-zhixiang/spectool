#ifndef __MATHFUNC_H__
#define __MATHFUNC_H__

#include <type_traits>

template <typename Iter>
auto mean(Iter begin, Iter end){
    typename std::iterator_traits<Iter>::value_type out = 0;
    int size = 0;
    while (begin != end)
    {
        out += *(begin++);
        size++;
    }
    return out / size;
}

template <typename Iter>
auto standarddif(Iter begin, Iter end){
    auto meanval = mean(begin, end);
    typename std::iterator_traits<Iter>::value_type out = 0;
    int size = 0;
    while (begin != end)
    {
        out += (*begin - meanval)*(*begin - meanval);
        begin++;
        size++;
    }
    return std::sqrt(out/size);
}


#endif // !__MATHFUNC_H__