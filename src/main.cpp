#include "perceptron.h"
#include <iostream>

Node node{3};

Layer layer{3, 3};


int main()
{
    srand(time(NULL));
    // FloatArray arr{{0.5, 0.5, 0.5}, 3};
    // FloatArray arr2{{0.5, 0.6, 0.7}, 3};
    // float x = node.forward_pass(arr);
    // FloatArray x2 = node.backwards_pass(arr, arr);
    // std::cout << x << '\n';
    // std::cout << x2.arr[1] << '\n';
    // std::cout << layer._nodes.arr[2]._weights.arr[2] << '\n';
    // FloatArray x3 = layer.forward_pass(arr);
    // std::cout << x3.arr[0];
    // FloatMatrix matrix{{arr, arr, arr}, 3};
    // FloatMatrix x4 = layer.backwards_pass(arr, matrix);
    // std::cout << x4.arr[0].arr[0];

    constexpr int TRAIN_DATA_SZ = 4;
    FloatArray x[TRAIN_DATA_SZ] = {
        {{1, 1}, 2},
        {{1, 0}, 2},
        {{0, 1}, 2},
        {{0, 0}, 2}
    };

    MLP mlp{};

    std::cout << mlp._layer_o._nodes.arr[0]._weights.arr[0] << '\n';

    std::cout << mlp._layer_h1._nodes.arr[2]._weights.arr[0] << '\n';

    std::cout << mlp._layer_h1._nodes.arr[1]._weights.arr[0] << '\n';

    float y[TRAIN_DATA_SZ] = {1, 0, 0, 1};

    constexpr int EPOCHS = 3000;

    float lowest_cost = 1000;
    for (int i{0}; i<EPOCHS; ++i)
    {
        float cost = 0;
        for (int j{0}; j<TRAIN_DATA_SZ; ++j)  
        {
            mlp.forward_pass(x[j], y[j]);
            mlp.backwards_pass(x[j], y[j]);
            cost += mlp.get_cost();
        }
        cost = cost / TRAIN_DATA_SZ;
        if (cost < lowest_cost)
            lowest_cost = cost;
        std::cout << cost << '\n';
    }
    std::cout << '\n' << "lowest cost: " << lowest_cost;

    return 1;
}
