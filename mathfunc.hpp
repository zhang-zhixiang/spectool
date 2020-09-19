#ifndef __MATHFUNC_H__
#define __MATHFUNC_H__

#include <type_traits>

template <typename Iter>
inline auto mean(Iter begin, Iter end){
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
inline auto standarddif(Iter begin, Iter end){
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

template <typename Iter, typename IterB>
inline auto normalize_arr(Iter begin, Iter end, IterB out){
    auto meanval = mean(begin, end);
    auto stdval = standarddif(begin, end);
    while (begin != end)
        *(out++) = (*(begin++) - meanval) / stdval;
}


#endif // !__MATHFUNC_H__