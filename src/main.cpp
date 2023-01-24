#include "perceptron.h"
#include <iostream>

Node node{3};

Layer layer{3, 3};

int main()
{
    FloatArray arr{{0.5, 0.5, 0.5}, 3};
    FloatArray arr2{{0.5, 0.6, 0.7}, 3};
    float x = node.forward_pass(arr);
    FloatArray x2 = node.backwards_pass(arr, arr);
    std::cout << x << '\n';
    std::cout << x2.arr[1] << '\n';
    std::cout << layer._nodes.arr[2]._weights.arr[2] << '\n';
    FloatArray x3 = layer.forward_pass(arr);
    std::cout << x3.arr[0];
    FloatMatrix matrix{{arr, arr, arr}, 3};
    FloatMatrix x4 = layer.backwards_pass(arr, matrix);
    std::cout << x4.arr[0].arr[0];

    return 1;
}
