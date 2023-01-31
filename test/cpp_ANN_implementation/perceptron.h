#include <cstdint>
#include <math.h>

constexpr uint8_t NUM_LAYERS = 3;
constexpr uint8_t MAX_NODES = 3;
constexpr uint8_t MAX_BRIGHTNESS = 254;
constexpr uint8_t MIN_BRIGHTNESS = 9;

struct FloatArray
{
    float arr[MAX_NODES];
    uint8_t size = 0;
};

struct FloatMatrix
{
    FloatArray arr[MAX_NODES];
    uint8_t size = 0;
};

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
    float forward_pass(const FloatArray& inputs);
    FloatArray backwards_pass(const FloatArray& inputs, const FloatArray& output_grads);
    const float& get_output() const { return _prev_output; }
    const FloatArray& get_weights() const { return _weights; }
    const FloatArray& get_inputs() const { return _prev_inputs; }
private:
    float _learning_rate;
    float _bias;
    float _prev_output;
    FloatArray _prev_inputs;
    FloatArray _weights;
};

struct NodeArray
{
    Node arr[MAX_NODES];
    uint8_t size = 0;
};

class Layer
{
public:
    Layer(const uint8_t& n_nodes, const uint8_t& n_inputs);
    void init_weights();
    FloatArray forward_pass(const FloatArray& inputs);
    FloatMatrix backwards_pass(const FloatArray& inputs,
                               const FloatMatrix& output_grad_matrix);
    const FloatArray& get_outputs() const { return _prev_outputs; }
    const NodeArray& get_nodes() const { return _nodes; }
private:
    NodeArray _nodes;
    FloatArray _prev_outputs;
};

class MLP
{
public:
    MLP();
    void init_weights();
    void forward_pass(const FloatArray& x, const float& y);
    void backwards_pass(const FloatArray& x, const float& y);
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
float min_max_scale(const float& x, const float& x_min, const float& x_max);
float brightness_scale(const float& x);
MinMaxValues get_min_max_values(const MLP& mlp);
