// #include "perceptron.h"
// #include <iostream>
// #include <random>

Node::Node(const uint8_t& n_inputs)
    : _prev_output{0}, _learning_rate{0.001}, _bias{0}, _weights{{}, n_inputs}
{
}

void Node::init_weights()
{
    _bias = random_decimal();
    for (uint8_t i{0}; i < _weights.size; ++i)
        _weights.arr[i] = random_decimal();
}

float Node::forward_pass(const StaticVec<float, MAX_NODES>& inputs)
{
    float z_sum = 0;
    for (uint8_t i{0}; i < _weights.size; ++i)
        z_sum += _weights.arr[i] * inputs.arr[i];
    z_sum += _bias;
    _prev_output = sigmoid(z_sum);
    _prev_inputs = inputs;
    return _prev_output;
}

StaticVec<float, MAX_NODES> Node::backwards_pass(const StaticVec<float, MAX_NODES>& inputs,
                                const StaticVec<float, MAX_NODES>& output_grads)
{
    StaticVec<float, MAX_NODES> input_grads;
    input_grads.size = 0;
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

StaticVec<float, MAX_NODES> Layer::forward_pass(const StaticVec<float, MAX_NODES>& inputs)
{
    _prev_outputs.size = 0;
    for (uint8_t i{0}; i < _nodes.size; ++i)
        _prev_outputs.arr[_prev_outputs.size++] =
            _nodes.arr[i].forward_pass(inputs);
    return _prev_outputs;
}

StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES> Layer::backwards_pass(const StaticVec<float, MAX_NODES>& inputs,
                                  const StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES>& output_grad_matrix)
{
    StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES> input_grad_matrix{{},0};
    input_grad_matrix.arr[0].size = 0;
    input_grad_matrix.arr[1].size = 0; // can't do aggregate initialization GCC 5.9...
    input_grad_matrix.arr[2].size = 0;
    for (uint8_t i{0}; i < _nodes.size; ++i)
    {
        StaticVec<float, MAX_NODES> input_grads =
            _nodes.arr[i].backwards_pass(inputs, output_grad_matrix.arr[i]);
        for (uint8_t j{0}; j < inputs.size; ++j)
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

void MLP::forward_pass(const StaticVec<float, MAX_NODES>& x, const float& y)
{
    StaticVec<float, MAX_NODES> output = _layer_h1.forward_pass(x);
    output = _layer_h2.forward_pass(output);
    output = _layer_o.forward_pass(output);
    _prev_cost = half_mse(output.arr[0], y);
}

void MLP::backwards_pass(const StaticVec<float, MAX_NODES>& x, const float& y)
{
    StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES> 
    out_output_grads{{StaticVec<float, MAX_NODES>{
    {-(y - _layer_o.get_outputs().arr[0])}, 1}}, 1};
    StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES> h2_output_grads =
        _layer_o.backwards_pass(_layer_h2.get_outputs(), out_output_grads);
    StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES> h1_output_grads =
        _layer_h2.backwards_pass(_layer_h1.get_outputs(), h2_output_grads);
    _layer_h1.backwards_pass(x, h1_output_grads);
}

const Layer& MLP::get_layer(uint8_t l) const
{
    switch (l)
    {
    case 0:
        return _layer_h1;
    case 1:
        return _layer_h2;
    case 2:
        return _layer_o;
    default:
        return _layer_o; // !!!!
    }
}

float sigmoid(const float& z) { return 1 / (1 + exp(-z)); }

float half_mse(const float& a, const float& y) { return 0.5 * pow((a - y), 2); }

float random_decimal() { return (((float)random(200)) / 100) - 1; }

float min_max_scale(const float& x, const float& x_min, const float& x_max)
{
    return (x - x_min) / (x_max - x_min);
}

float brightness_scale(const float& x)
// assume values range between 0-1
{
    // Serial.print("min maxed = ");
    // Serial.println(x);
    return (x * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)) + 9;
}

MinMaxValues get_min_max_values(const MLP& mlp)
{
    MinMaxValues values;
    const Node& temp = mlp.get_layer(0).get_nodes().arr[0];
    values.node_min = temp.get_output();
    values.node_max = temp.get_output();
    values.link_min = temp.get_weights().arr[0] * temp.get_inputs().arr[0];
    values.link_max = temp.get_weights().arr[0] * temp.get_inputs().arr[0];

    for (int i{0}; i < NUM_LAYERS; ++i)
    {
        const StaticVec<Node, MAX_NODES>& nodes = mlp.get_layer(i).get_nodes();
        for (int j{0}; j < nodes.size; ++j)
        {
            const Node& node = nodes.arr[j];
            if (node.get_output() > values.node_max)
                values.node_max = node.get_output();
            if (node.get_output() < values.node_min)
                values.node_min = node.get_output();
            for (int k{0}; k < node.get_weights().size; ++k)
            {
                float link_strength =
                    node.get_weights().arr[k] * node.get_inputs().arr[k];
                if (link_strength > values.link_max)
                    values.link_max = link_strength;
                if (link_strength < values.link_min)
                    values.link_min = link_strength;
            }
        }
    }
    return values;
}
