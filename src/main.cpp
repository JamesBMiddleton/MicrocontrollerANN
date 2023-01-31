#include "data.h"
#include <iostream>

float min_max_scale(const float& x, const float& x_min, const float& x_max)
{
    return (x - x_min) / (x_max - x_min);
}

float brightness_scale(const float& x)
// assume values range between 0-1
{
    return (x * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)) + 9;
}

MLP mlp{};

int main()
{
    srand(time(NULL));
    mlp.init_weights();

    constexpr int EPOCHS = 1;

    float lowest_cost = 1000;
    for (int i{0}; i<EPOCHS; ++i)
    {
        float cost = 0;
        for (int j{0}; j<20; ++j)  
        {
            mlp.forward_pass(x_train[j], y_train[j]);
            float scaled = min_max_scale(x_train[j].arr[0], X0_TRAIN_MIN, X0_TRAIN_MAX);
            std::cout << brightness_scale(scaled) << '\n';

            mlp.backwards_pass(x_train[j], y_train[j]);
            cost += mlp.get_cost();
        }
        if (cost < lowest_cost)
            lowest_cost = cost;
        // std::cout << cost << '\n';
    }
    std::cout << '\n' << "lowest cost: " << lowest_cost;

    return 1;
}
