#include <cstdint>
#include <math.h>

constexpr uint8_t MAX_NODES = 3;

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

class Node
{
public:
    Node(){}; // bad practice?
    Node(const uint8_t& n_inputs);
    float forward_pass(FloatArray inputs); // const ref?
    FloatArray backwards_pass(FloatArray inputs, FloatArray output_grads);
    // private:
    float _prev_output;
    float _learning_rate;
    float _bias;
    FloatArray _weights;
};

class Layer
{
public:
    Layer(const uint8_t& n_nodes, const uint8_t& n_inputs);
    FloatArray forward_pass(FloatArray inputs);
    FloatMatrix backwards_pass(FloatArray inputs,
                               FloatMatrix output_grad_matrix);
    // private:
    struct NodeArray
    {
        Node arr[MAX_NODES];
        uint8_t size = 0;
    };
    NodeArray _nodes;
    FloatArray _prev_outputs;
};

class MLP
{
public:
    MLP();
    void forward_pass(FloatArray x, float y);
    void backwards_pass(FloatArray x, float y);
    const float& get_cost() const { return _prev_cost; }
    // private:
    Layer _layer_h1;
    Layer _layer_h2;
    Layer _layer_o;
    float _prev_cost;
};

float sigmoid(const float& z);
float half_mse(const float& a, const float& b);
float random_decimal();
