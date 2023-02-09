#include <Adafruit_Protomatter.h>
#include <Fonts/Picopixel.h>
#include "pulsar.h"
#include "perceptron.h"
#include "data.h"
#include "static_vec.h"

// #define DEBUG

#ifdef DEBUG
String error = "";
#endif

uint8_t rgbPins[]  = {7, 8, 9, 10, 11, 12};
uint8_t addrPins[] = {17, 18, 19, 20};
uint8_t instancePin   = 14;
uint8_t latchPin   = 15;
uint8_t oePin      = 16;

constexpr uint8_t btn_upPin = 2;
bool show_loss = true; // up button toggles loss

constexpr uint8_t btn_downPin = 3;
uint8_t btn_downState = HIGH;
bool color_links = false; // down buttons colours links (no toggle)

uint8_t matrix_width = 64; // total width
uint8_t colour_depth = 5; // colour bit depth 1-6
uint8_t matrix_num = 1; // number of matrices
uint8_t row_addr_lines = 4; // matrix height inferred from this
bool double_buffer = true; // improve animation smoothness for extra ram

Adafruit_Protomatter matrix{matrix_width, colour_depth, matrix_num, rgbPins, 
                            row_addr_lines, addrPins, instancePin, latchPin,
                            oePin, double_buffer};

constexpr uint8_t center = 15;
constexpr uint8_t col0 = 4;
constexpr uint8_t col1 = 22;
constexpr uint8_t col2 = 46;
constexpr uint8_t col3 = 58;

StaticVec<StaticVec<NodePulsar, MAX_NODES>, NUM_LAYERS> node_matrix;
StaticVec<StaticVec<StaticVec<LinkPulsar, MAX_LINKS>, MAX_NODES>, NUM_LAYERS> link_matrix;
NodePulsar input1_node{col0, center-6, 2};
NodePulsar input2_node{col0, center+6, 2};


MLP mlp{};


// ----- Helper functions for globals ------ //

void setup_matrix_library()
{
    ProtomatterStatus status = matrix.begin();
    Serial.print("Protomatter begin() status: ");
    Serial.println((int)status);
    if(status != PROTOMATTER_OK)
        for(;;);

    matrix.setFont(&Picopixel);
    matrix.setTextColor(matrix.color565(16,16,16));
}

void construct_pulsar_matrices()
{
    node_matrix.push_back(StaticVec<NodePulsar, MAX_NODES>{3});
    node_matrix[0][0] = NodePulsar{col1, center-12, 2};
    node_matrix[0][1] = NodePulsar{col1, center, 2};
    node_matrix[0][2] = NodePulsar{col1, center+12, 2};

    node_matrix.push_back(StaticVec<NodePulsar, MAX_NODES>{3});
    node_matrix[1][0] = NodePulsar{col2, center-12, 2};
    node_matrix[1][1] = NodePulsar{col2, center, 2};
    node_matrix[1][2] = NodePulsar{col2, center+12, 2};

    node_matrix.push_back(StaticVec<NodePulsar, MAX_NODES>{1});
    node_matrix[2][0] = NodePulsar{col3, center, 2};


    link_matrix.push_back(StaticVec<StaticVec<LinkPulsar, MAX_LINKS>, MAX_NODES>{3});
    link_matrix[0][0] = StaticVec<LinkPulsar, MAX_LINKS>{2};
    link_matrix[0][1] = StaticVec<LinkPulsar, MAX_LINKS>{2};
    link_matrix[0][2] = StaticVec<LinkPulsar, MAX_LINKS>{2};

    link_nodes(&input1_node, &node_matrix[0][0], &link_matrix[0][0][0]);
    link_nodes(&input2_node, &node_matrix[0][0], &link_matrix[0][0][1]);

    link_nodes(&input1_node, &node_matrix[0][1], &link_matrix[0][1][0]);
    link_nodes(&input2_node, &node_matrix[0][1], &link_matrix[0][1][1]);

    link_nodes(&input1_node, &node_matrix[0][2], &link_matrix[0][2][0]);
    link_nodes(&input2_node, &node_matrix[0][2], &link_matrix[0][2][1]);

    link_matrix.push_back(StaticVec<StaticVec<LinkPulsar, MAX_LINKS>, MAX_NODES>{3});
    link_matrix[1][0] = StaticVec<LinkPulsar, MAX_LINKS>{3};
    link_matrix[1][1] = StaticVec<LinkPulsar, MAX_LINKS>{3};
    link_matrix[1][2] = StaticVec<LinkPulsar, MAX_LINKS>{3};

    link_nodes(&node_matrix[0][0], &node_matrix[1][0], &link_matrix[1][0][0]);
    link_nodes(&node_matrix[0][1], &node_matrix[1][0], &link_matrix[1][0][1]);
    link_nodes(&node_matrix[0][2], &node_matrix[1][0], &link_matrix[1][0][2]);

    link_nodes(&node_matrix[0][0], &node_matrix[1][1], &link_matrix[1][1][0]);
    link_nodes(&node_matrix[0][1], &node_matrix[1][1], &link_matrix[1][1][1]);
    link_nodes(&node_matrix[0][2], &node_matrix[1][1], &link_matrix[1][1][2]);

    link_nodes(&node_matrix[0][0], &node_matrix[1][2], &link_matrix[1][2][0]);
    link_nodes(&node_matrix[0][1], &node_matrix[1][2], &link_matrix[1][2][1]);
    link_nodes(&node_matrix[0][2], &node_matrix[1][2], &link_matrix[1][2][2]);

    link_matrix.push_back(StaticVec<StaticVec<LinkPulsar, MAX_LINKS>, MAX_NODES>{1});
    link_matrix[2][0] = StaticVec<LinkPulsar, MAX_LINKS>{3};

    link_nodes(&node_matrix[1][0], &node_matrix[2][0], &link_matrix[2][0][0]);
    link_nodes(&node_matrix[1][1], &node_matrix[2][0], &link_matrix[2][0][1]);
    link_nodes(&node_matrix[1][2], &node_matrix[2][0], &link_matrix[2][0][2]);
}

void toggle_pulsar_colours()
{
    static bool color = true;
    color = !color;
    for (int i{0}; i<node_matrix.size(); ++i)
    {
        const StaticVec<Node, MAX_NODES>& nodes = mlp.get_layer(i).get_nodes();
        for (int j{0}; j<node_matrix[i].size(); ++j)
        {
            const Node& node = nodes[j];
            for (int k{0}; k<link_matrix[i][j].size(); ++k)
            {
                if (node.get_weights()[k] >= 0)
                    link_matrix[i][j][k].set_hue(HSV_RED);
                else
                    link_matrix[i][j][k].set_hue(HSV_BLUE);
                if (color)
                    link_matrix[i][j][k].set_sat(255);
                else
                    link_matrix[i][j][k].set_sat(0);
            }
        }
    }
}

void update_pulsar_brightness_forward(const StaticVec<float, MAX_NODES> inputs)
{
    MinMaxValues values = get_abs_minmaxes_forward(mlp);

    float brightness = minmax_scale(abs(inputs[0]), X0_TRAIN_MIN, X0_TRAIN_MAX, MIN_BRIGHTNESS, MAX_BRIGHTNESS);
    input1_node.set_max_brightness(brightness);
    brightness = minmax_scale(abs(inputs[1]), X1_TRAIN_MIN, X1_TRAIN_MAX, MIN_BRIGHTNESS, MAX_BRIGHTNESS);
    input2_node.set_max_brightness(brightness);
    for (int i{0}; i<node_matrix.size(); ++i)
    {
        const StaticVec<Node, MAX_NODES>& nodes = mlp.get_layer(i).get_nodes();
        for (int j{0}; j<node_matrix[i].size(); ++j)
        {
            const Node& node = nodes[j];
            float output = node.get_output();
            brightness = minmax_scale(output, values.node_min, values.node_max, MIN_BRIGHTNESS, MAX_BRIGHTNESS);
            node_matrix[i][j].set_max_brightness(brightness);
            for (int k{0}; k<link_matrix[i][j].size(); ++k)
            {
                float link_strength = node.get_weights()[k] * node.get_inputs()[k];
                brightness = minmax_scale(abs(link_strength), values.link_min, values.link_max, MIN_BRIGHTNESS, MAX_BRIGHTNESS);
                link_matrix[i][j][k].set_max_brightness(brightness);
                // Serial.println(link_matrix[i][j][k].get_max_brightness());
            }
        }
    }
}

void update_pulsar_brightness_backward()
{
    MinMaxValues values = get_abs_minmaxes_backward(mlp);
    input1_node.set_max_brightness(MIN_BRIGHTNESS);
    input2_node.set_max_brightness(MIN_BRIGHTNESS);
    for (int i{0}; i<node_matrix.size(); ++i)
    {
        const StaticVec<Node, MAX_NODES>& nodes = mlp.get_layer(i).get_nodes();
        for (int j{0}; j<node_matrix[i].size(); ++j)
        {
            const Node& node = nodes[j];
            float grad_sum = 0;
            for (int k{0}; k<link_matrix[i][j].size(); ++k)
            {
                float weight_grad = node.get_weight_grads()[k];
                float brightness = minmax_scale(abs(weight_grad), values.link_min, values.link_max, MIN_BRIGHTNESS, MAX_BRIGHTNESS);
                link_matrix[i][j][k].set_max_brightness(brightness);
                grad_sum += weight_grad;
            }
            float brightness = minmax_scale(abs(grad_sum), values.node_min, values.node_max, MIN_BRIGHTNESS, MAX_BRIGHTNESS);
            node_matrix[i][j].set_max_brightness(brightness);
        }
    }
}


void update_draw_pulsars()
{
    for (int i{0}; i<link_matrix.size(); ++i)
        for (int j{0}; j<link_matrix[i].size(); ++j)
            for (int k{0}; k<link_matrix[i][j].size(); ++k)
            {
                link_matrix[i][j][k].update();
                link_matrix[i][j][k].draw();
            }
    input1_node.update();
    input1_node.draw();
    input2_node.update();
    input2_node.draw();
    for (int i{0}; i<node_matrix.size(); ++i)
        for (int j{0}; j<node_matrix[i].size(); ++j)
        {
            node_matrix[i][j].update();
            node_matrix[i][j].draw();
        }
}

void shuffle_data(StaticVec<StaticVec<float, MAX_NODES>, TRAIN_DATA_SZ>& X,
StaticVec<float, TRAIN_DATA_SZ>& y)
{
    for (int i{0}; i < y.size(); ++i)
    {
        int random_index = rand() % y.size();
        for (int j{0}; j < X[i].size(); ++j)
        {
            float temp = X[i][j];
            X[i][j] = X[random_index][j];
            X[random_index][j] = temp;
        }
        float temp = y[i];
        y[i] = y[random_index];
        y[random_index] = temp;
    }
}

void check_buttons()
{
    static uint delay = 0;
    if (delay)
        --delay;
    else
    {
        if (digitalRead(btn_upPin) == LOW)
        {
            matrix.fillRect(50, 26, 14, 5, 0);
            show_loss = !show_loss;
            delay = 100;
        }
        if (digitalRead(btn_downPin) == LOW)
        {
            toggle_pulsar_colours();
            delay = 100;
        }
    }
}


void print_cost(const float& cost)
{
    static float lowest_cost = 100000; // !
    if (cost < lowest_cost)
        lowest_cost = cost;
    if (show_loss)
    {
        matrix.fillRect(51, 27, 14, 5, 0);
        matrix.setCursor(51, 31);
        matrix.print(lowest_cost);
    }
}


// ----- Main setup and loop ------ //

void setup() {

    Serial.begin(9600); 

    setup_matrix_library();
    pinMode(btn_upPin, INPUT_PULLUP);
    pinMode(btn_downPin, INPUT_PULLUP);

    x_train_matrix = populate_x_train(raw_x_train);
    y_train_vec = populate_y_train(raw_y_train);

    randomSeed(analogRead(0));
    mlp.init_weights();
    construct_pulsar_matrices();
}


void loop() {
    check_buttons();
    static float cost = 0;
    static uint instance = 0;
    static int i = 0;
    if (i++ == 1000)
    {
        i = 0;
        ++instance;
        if (instance == TRAIN_DATA_SZ)
        {
            instance = 0;
            shuffle_data(x_train_matrix, y_train_vec);
        }

        mlp.forward_pass(x_train_matrix[instance], y_train_vec[instance]);
        update_pulsar_brightness_forward(x_train_matrix[instance]);
        mlp.backwards_pass(x_train_matrix[instance], y_train_vec[instance]);
        cost += mlp.get_cost();


        if (instance % BATCH_SIZE == 0)
        {
            print_cost(cost * 100); // *100 for more precision in viz
            cost = 0;
            update_pulsar_brightness_backward();
            node_matrix[2][0].init_b_pulse();
        }
        else
        {
            input1_node.init_f_pulse();
            input2_node.init_f_pulse();
        }
    }

    update_draw_pulsars();

    matrix.show();


#ifdef DEBUG
    if (error != "")
        Serial.println(error);
#endif
}
