
Node::Node(const uint8_t& n_inputs)
    :_learning_rate{0.01}, _lr_decay_counter{0}, 
    _lr_decay_threshold{TRAIN_DATA_SZ}, _bias{0}, _prev_output{0},
    _weight_grads{n_inputs}, _prev_weight_avg_grads{n_inputs}, _bias_grads{}
{
    for (int i{0}; i < n_inputs; ++i)
        _weights.push_back(0);
}

void Node::init_weights()
{
    _bias = random_decimal();
    for (uint8_t i{0}; i < _weights.size(); ++i)
        _weights[i] = random_decimal();
}

void Node::update_learning_rate()
// half the learning rate every epoch
// simple but seems to work better than exponential decay?
{
    ++_lr_decay_counter;
    if (_lr_decay_counter == _lr_decay_threshold)
    {
        _learning_rate = _learning_rate / 2;
        _lr_decay_counter = 0;
    }
}

void Node::take_step()
{
    for (int i{0}; i < _weight_grads.size(); ++i) 
    {
        float avg_weight_grad = 0;
        for (int j{0}; j < _weight_grads[i].size(); ++j)
            avg_weight_grad += _weight_grads[i][j];
        avg_weight_grad = avg_weight_grad / (float)_weight_grads[i].size();
        _prev_weight_avg_grads[i] = avg_weight_grad;
        _weights[i] = _weights[i] - (_learning_rate * avg_weight_grad);
        _weight_grads[i].clear();
    }
    float avg_bias_grad = 0;
    for (int i{0}; i < _bias_grads.size(); ++i)
        avg_bias_grad += _bias_grads[i];
    avg_bias_grad = avg_bias_grad / _bias_grads.size();
    _bias = _bias - (_learning_rate * avg_bias_grad);
    _bias_grads.clear();
    this->update_learning_rate();
}


float Node::forward_pass(const StaticVec<float, MAX_NODES>& inputs)
{
    float z_sum = 0;
    for (uint8_t i{0}; i < _weights.size(); ++i)
        z_sum += _weights[i] * inputs[i];
    z_sum += _bias;
    _prev_output = sigmoid(z_sum);
    _prev_inputs = inputs;
    return _prev_output;
}

StaticVec<float, MAX_NODES> Node::backwards_pass(const StaticVec<float, MAX_NODES>& inputs,
                                const StaticVec<float, MAX_NODES>& output_grads)
{
    StaticVec<float, MAX_NODES> input_grads;
    float z_grad = _prev_output * (1 - _prev_output);
    for (uint8_t i{0}; i < _weights.size(); ++i)
    {
        float part_input_grad = z_grad * _weights[i];
        float full_input_grad = 0;
        float part_weight_grad = z_grad * inputs[i];
        float full_weight_grad = 0;
        for (uint8_t j{0}; j < output_grads.size(); ++j)
        {
            full_input_grad += output_grads[j] * part_input_grad;
            full_weight_grad += output_grads[j] * part_weight_grad;
        }
        input_grads.push_back(full_input_grad);
        _weight_grads[i].push_back(full_weight_grad);
    }
    float bias_grad = _prev_output * (1 - _prev_output);
    _bias_grads.push_back(bias_grad);
    if (_bias_grads.size() == _bias_grads.max_size())
        this->take_step();
    return input_grads;
}


Layer::Layer(const uint8_t& n_nodes, const uint8_t& n_inputs)
{
    for (uint8_t i{0}; i < n_nodes; ++i)
        _nodes.push_back(Node{n_inputs});
}

void Layer::init_weights()
{
    for (uint8_t i{0}; i < _nodes.size(); ++i)
        _nodes[i].init_weights();
}

StaticVec<float, MAX_NODES> Layer::forward_pass(const StaticVec<float, MAX_NODES>& inputs)
{
    _prev_outputs.clear();
    for (uint8_t i{0}; i < _nodes.size(); ++i)
        _prev_outputs.push_back(_nodes[i].forward_pass(inputs));
    return _prev_outputs;
}

StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES> Layer::backwards_pass(const StaticVec<float, MAX_NODES>& inputs,
                                  const StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES>& output_grad_matrix)
{
    StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES> input_grad_matrix{inputs.size()};
    for (uint8_t i{0}; i < _nodes.size(); ++i)
    {
        StaticVec<float, MAX_NODES> input_grads = _nodes[i].backwards_pass(inputs, output_grad_matrix[i]);
        for (uint8_t j{0}; j < inputs.size(); ++j)
            input_grad_matrix[j].push_back(input_grads[j]);
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
    _prev_cost = half_mse(output[0], y);
}

void MLP::backwards_pass(const StaticVec<float, MAX_NODES>& x, const float& y)
{
    StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES> out_output_grads;
    StaticVec<float, MAX_NODES> output;
    output.push_back(-(y - _layer_o.get_outputs()[0]));
    out_output_grads.push_back(output);
    StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES> h2_output_grads =
        _layer_o.backwards_pass(_layer_h2.get_outputs(), out_output_grads);
    StaticVec<StaticVec<float, MAX_NODES>, MAX_NODES> h1_output_grads =
        _layer_h2.backwards_pass(_layer_h1.get_outputs(), h2_output_grads);
    _layer_h1.backwards_pass(x, h1_output_grads);
}

const Layer& MLP::get_layer(uint8_t l) const
{
    if (l == 0)
        return _layer_h1;
    if (l == 1)
        return _layer_h2;
    if (l == 2)
        return _layer_o;
#ifdef DEBUG
    error = "Invalid layer requested from get_layer().";
#endif
    return _layer_o; // !!!! 
}

float sigmoid(const float& z) { return 1 / (1 + exp(-z)); }

float half_mse(const float& a, const float& y) { return 0.5 * pow((a - y), 2); }

float random_decimal() { return (((float)random(200)) / 100) - 1; }

float minmax_scale(const float& x, const float& x_min, const float& x_max, const float& new_min, const float& new_max)
{
    return MIN_BRIGHTNESS + (((x - x_min) * (new_max - new_min)) / (x_max - x_min));
}

MinMaxValues get_abs_minmaxes_forward(const MLP& mlp)
// absolute min_max values for forward pass
// 'node' = z_sum, 'link' = input * weight
{
    MinMaxValues values;
    const Node& temp = mlp.get_layer(0).get_nodes()[0];
    values.node_min = abs(temp.get_output());
    values.node_max = abs(temp.get_output());
    values.link_min = abs(temp.get_weights()[0] * temp.get_inputs()[0]);
    values.link_max = abs(temp.get_weights()[0] * temp.get_inputs()[0]);

    for (int i{0}; i < NUM_LAYERS; ++i)
    {
        const StaticVec<Node, MAX_NODES>& nodes = mlp.get_layer(i).get_nodes();
        for (int j{0}; j < nodes.size(); ++j)
        {
            const Node& node = nodes[j];
            if (node.get_output() > values.node_max)
                values.node_max = node.get_output();
            if (node.get_output() < values.node_min)
                values.node_min = node.get_output();
            for (int k{0}; k < node.get_weights().size(); ++k)
            {
                float link_strength =
                    abs(node.get_weights()[k] * node.get_inputs()[k]);
                if (link_strength > values.link_max)
                    values.link_max = link_strength;
                if (link_strength < values.link_min)
                    values.link_min = link_strength;
            }
        }
    }
    return values;
}

MinMaxValues get_abs_minmaxes_backward(const MLP& mlp)
// absolute min_max values for back pass
// 'node' = sum of weight_grads, 'link' = weight_grads
{
    MinMaxValues values;
    const Node& temp = mlp.get_layer(0).get_nodes()[0];
    values.link_min = abs(temp.get_weight_grads()[0]);
    values.link_max = abs(temp.get_weight_grads()[0]);
    float grad_sum = 0;
    for (int i{0}; i < temp.get_weight_grads().size(); ++i)
        grad_sum += temp.get_weight_grads()[i];
    values.node_min = abs(grad_sum);
    values.node_max = abs(grad_sum);

    for (int i{0}; i < NUM_LAYERS; ++i)
    {
        const StaticVec<Node, MAX_NODES>& nodes = mlp.get_layer(i).get_nodes();
        for (int j{0}; j < nodes.size(); ++j)
        {
            const Node& node = nodes[j];
            float grad_sum = 0;
            for (int k{0}; k < node.get_weights().size(); ++k)
            {
                float weight_grad = abs(node.get_weight_grads()[k]);
                if (weight_grad > values.link_max)
                    values.link_max = weight_grad;
                if (weight_grad < values.link_min)
                    values.link_min = weight_grad;
                grad_sum += weight_grad;
            }
            if (grad_sum > values.node_max)
                values.node_max = grad_sum;
            if (grad_sum < values.node_min)
                values.node_min = grad_sum;
        }
    }
    return values;
}
