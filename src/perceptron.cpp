#include "perceptron.h"
#include <iostream>
#include <random>

Node::Node(const uint8_t& n_inputs)
    : _prev_output{0}, _learning_rate{0.001}, _bias{0}, _weights{{},n_inputs}
{}

void Node::init_weights()
{
    _bias = random_decimal();
    for (uint8_t i{0}; i<_weights.size; ++i)
        _weights.arr[i] = random_decimal();
}

float Node::forward_pass(const FloatArray& inputs)
{
    // if (inputs.size != _weights.size)
    // Serial.println("ERROR: inputs != weights");

    float z_sum = 0;
    for (uint8_t i{0}; i < _weights.size; ++i)
        z_sum += _weights.arr[i] * inputs.arr[i];
    z_sum += _bias;
    _prev_output = sigmoid(z_sum);
    return _prev_output;
}

FloatArray Node::backwards_pass(const FloatArray& inputs, const FloatArray& output_grads)
{
    // if (inputs.size != _weights.size)
    // Serial.println("ERROR: inputs != weights");

    FloatArray input_grads;
    float z_grad = _prev_output * (1 - _prev_output);
    for (uint8_t i{0}; i < _weights.size; ++i)
    {
        float part_input_grad = z_grad * _weights.arr[i];
        float full_input_grad = 0;
        float part_weight_grad = z_grad * inputs.arr[i];
        float full_weight_grad = 0;
        for (uint8_t j{0}; j < output_grads.size; ++j)
        {
            full_input_grad += output_grads.arr[j] * part_input_grad;
            full_weight_grad += output_grads.arr[j] * part_weight_grad;
        }
        input_grads.arr[input_grads.size++] = full_input_grad;
        _weights.arr[i] = _weights.arr[i] - (_learning_rate * full_weight_grad);
    }
    float bias_grad = _prev_output * (1 - _prev_output);
    _bias = _bias - (_learning_rate * bias_grad);
    return input_grads;
}

Layer::Layer(const uint8_t& n_nodes, const uint8_t& n_inputs)
{
    for (uint8_t i{0}; i < n_nodes; ++i)
        _nodes.arr[_nodes.size++] = Node{n_inputs};
}

void Layer::init_weights()
{
    for (uint8_t i{0}; i < _nodes.size; ++i)
        _nodes.arr[i].init_weights(); 
}

FloatArray Layer::forward_pass(const FloatArray& inputs)
{
    _prev_outputs.size = 0;
    for (uint8_t i{0}; i < _nodes.size; ++i)
        _prev_outputs.arr[_prev_outputs.size++] =
            _nodes.arr[i].forward_pass(inputs);
    return _prev_outputs;
}

FloatMatrix Layer::backwards_pass(const FloatArray& inputs,
                                  const FloatMatrix& output_grad_matrix)
{
    FloatMatrix input_grad_matrix;
    input_grad_matrix.size = inputs.size;
    for (uint8_t i{0}; i < _nodes.size; ++i)
    {
        FloatArray input_grads =
            _nodes.arr[i].backwards_pass(inputs, output_grad_matrix.arr[i]);
        for (uint8_t j{0}; j < input_grad_matrix.size; ++j)
            input_grad_matrix.arr[j].arr[input_grad_matrix.arr[j].size++] =
                input_grads.arr[j];
    }
    return input_grad_matrix;
}

MLP::MLP() : _layer_h1{3, 2}, _layer_h2{3, 3}, _layer_o{1, 3}, _prev_cost{0} {}

void MLP::init_weights()
{
    _layer_h1.init_weights();
    _layer_h2.init_weights();
    _layer_o.init_weights();
}

void MLP::forward_pass(const FloatArray& x, const float& y)
{
    FloatArray output = _layer_h1.forward_pass(x);
    output = _layer_h2.forward_pass(output);
    output = _layer_o.forward_pass(output);
    _prev_cost = half_mse(output.arr[0], y);
}

void MLP::backwards_pass(const FloatArray& x, const float& y)
{
    FloatMatrix out_output_grads{{{{-(y - _layer_o.get_outputs().arr[0])}, 1}},
                                 1};
    FloatMatrix h2_output_grads =
        _layer_o.backwards_pass(_layer_h2.get_outputs(), out_output_grads);
    FloatMatrix h1_output_grads =
        _layer_h2.backwards_pass(_layer_h1.get_outputs(), h2_output_grads);
    _layer_h1.backwards_pass(x, h1_output_grads);
}

float sigmoid(const float& z) { return 1 / (1 + exp(-z)); }

float half_mse(const float& a, const float& y) { return 0.5 * pow((a - y), 2); }

float random_decimal() { return (float)(rand() % 100) / 100; }
