
template<typename T, int N>
void StaticVec<T, N>::push_back(const T& value)
{
  #ifdef DEBUG
  if (_size == max_size())
    error = "StaticVec Array Overflow!";
  #endif
   _arr[_size++] = value;
}

template<typename T, int N>
const T& StaticVec<T, N>::operator[](int i) const
{
  #ifdef DEBUG
  if (i >= _size)
    error = "StaticVec Subscript Out Of Range!";
  #endif
  return _arr[i];
}

template<typename T, int N>
T& StaticVec<T, N>::operator[](int i)
{
  #ifdef DEBUG
  if (i >= _size)
    error = "StaticVec Subscript Out Of Range!";
  #endif
  return _arr[i];
}


