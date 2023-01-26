#include "data.h"
#include <iostream>

MLP mlp{};

int main()
{
    srand(time(NULL));
    mlp.init_weights();

    // std::cout << mlp._layer_o._nodes.arr[0]._weights.arr[0] << '\n';
    //
    // std::cout << mlp._layer_h1._nodes.arr[2]._weights.arr[0] << '\n';
    //
    // std::cout << mlp._layer_h1._nodes.arr[1]._weights.arr[0] << '\n';

    constexpr int EPOCHS = 30;

    float lowest_cost = 1000;
    for (int i{0}; i<EPOCHS; ++i)
    {
        float cost = 0;
        for (int j{0}; j<TRAIN_DATA_SZ; ++j)  
        {
            mlp.forward_pass(x_train[j], y_train[j]);
            mlp.backwards_pass(x_train[j], y_train[j]);
            cost += mlp.get_cost();
        }
        if (cost < lowest_cost)
            lowest_cost = cost;
        std::cout << cost << '\n';
    }
    std::cout << '\n' << "lowest cost: " << lowest_cost;

    return 1;
}
