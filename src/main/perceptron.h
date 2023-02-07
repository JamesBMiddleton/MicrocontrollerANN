#ifndef PERCEPTRON 
#define PERCEPTRON

#include <math.h>
#include "static_vec.h"

constexpr uint8_t NUM_LAYERS = 3;

struct MinMaxValues
// 'node' = z_sum, 'link' = input * weight
{
    float node_min;
    float node_max;
    float link_min;
    float link_max;
};

class Node
{
public:
    Node(){}; // bad practice?
    Node(const uint8_t& n_inputs);
    void init_weights();
    float forward_pass(const StaticVec<float, MAX_NODES>& inputs);
    StaticVec<float, MAX_NODES>
    backwards_pass(const StaticVec<float, MAX_NODES>& inputs,
                   const StaticVec<float, MAX_NODES>& output_grads);
    const float& get_output() const { return _prev_output; }
    const StaticVec<float, MAX_NODES>& get_weights() const { return _weights; }
    const StaticVec<float, MAX_NODES>& get_inputs() const
    {
        return _prev_inputs;
    }

private:
    float _learning_rate;
    float _bias;
    float _prev_output;
    StaticVec<float, MAX_NODES> _prev_inputs;
    StaticVec<float, MAX_NODES> _weights;
};

class Layer
{
public:
    Layer(const uint8_t& n_nodes, const uint8_t& n_inputs);
    void init_weights();
    StaticVec<float, MAX_NODES>
    forward_pass(const StaticVec<float, MAX_NODES>& inputs);
    StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES>
    backwards_pass(const StaticVec<float, MAX_NODES>& inputs,
                   const StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES>&
                       output_grad_matrix);
    const StaticVec<float, MAX_NODES>& get_outputs() const
    {
        return _prev_outputs;
    }
    const StaticVec<Node, MAX_NODES>& get_nodes() const { return _nodes; }

private:
    StaticVec<Node, MAX_NODES> _nodes;
    StaticVec<float, MAX_NODES> _prev_outputs;
};

class MLP
{
public:
    MLP();
    void init_weights();
    void forward_pass(const StaticVec<float, MAX_NODES>& x, const float& y);
    void backwards_pass(const StaticVec<float, MAX_NODES>& x, const float& y);
    const float& get_cost() const { return _prev_cost; }
    const Layer& get_layer(uint8_t l) const;

private:
    Layer _layer_h1;
    Layer _layer_h2;
    Layer _layer_o;
    float _prev_cost;
};

float sigmoid(const float& z);
float half_mse(const float& a, const float& b);
float random_decimal();
float minmax_scale(const float& x, const float& x_min, const float& x_max, const float& new_min, const float& new_max);
MinMaxValues get_abs_minmaxes(const MLP& mlp);

#endif
