#include "data.h"
#include <iostream>

MLP mlp{};

int main()
{
    srand(time(NULL));
    mlp.init_weights();

    // mlp.forward_pass(x_train.arr[0], y_train[0]);
    // std::cout << "check" << std::flush;
    // mlp.backwards_pass(x_train.arr[0], y_train[0]);

    constexpr int EPOCHS = 1;

    float lowest_cost = 1000;
    for (int i{0}; i < EPOCHS; ++i)
    {
        float cost = 0;
        for (int j{0}; j < 1; ++j)  
        {
            mlp.forward_pass(x_train.arr[j], y_train[j]);

            // float scaled = min_max_scale(x_train[j].arr[0], X0_TRAIN_MIN, X0_TRAIN_MAX);
            // std::cout << brightness_scale(scaled) << '\n';

            MinMaxValues v = get_min_max_values(mlp);
            std::cout << "link max = " << v.link_max << '\n';
            std::cout << "link min = " << v.link_min << '\n';
            std::cout << "node max = " << v.node_max << '\n';
            std::cout << "node min = " << v.node_min << '\n';


            mlp.backwards_pass(x_train.arr[j], y_train[j]);
            cost += mlp.get_cost();
        }
        if (cost < lowest_cost)
            lowest_cost = cost;
        // std::cout << cost << '\n';
    }
    std::cout << '\n' << "lowest cost: " << lowest_cost;

    return 1;
}
