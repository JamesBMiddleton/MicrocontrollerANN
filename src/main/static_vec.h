#ifndef STATIC_VEC
#define STATIC_VEC

#include <array>
#include <stdint.h>

constexpr int MAX_NODES = 3; // not clean

template <typename T, int N>
class StaticVec
{
public:
    StaticVec() {}
    StaticVec(int i) :_size{i} {}
    void push_back(const T& value);
    int size() const { return _size; }
    int max_size() const { return sizeof(_arr) / sizeof(T); }
    void clear() { _size = 0; }
    const T& operator[](int i) const;
    T& operator[](int i);
private:
    T _arr[N];
    int _size = 0;
};

#endif
