#ifndef STATIC_VEC
#define STATIC_VEC

#include <array>
#include <stdint.h>

constexpr int MAX_NODES = 3;

template <typename T, int N>
class StaticVec
{
public:
    StaticVec() {}
    StaticVec(int i) :_size{i} {}
    void push_back(const T& value);
    int size() const { return _size; }
    void clear() { _size = 0; }
    const T& operator[](int i) const;
    T& operator[](int i);
private:
    std::array<T, N> _arr{};
    int _size = 0;
};

#endif
