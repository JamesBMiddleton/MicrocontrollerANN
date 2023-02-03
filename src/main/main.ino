#include <Adafruit_Protomatter.h>
#include "pulsar.h"
#include "data.h"

uint8_t rgbPins[]  = {7, 8, 9, 10, 11, 12};
uint8_t addrPins[] = {17, 18, 19, 20};
uint8_t clockPin   = 14;
uint8_t latchPin   = 15;
uint8_t oePin      = 16;

uint8_t matrix_width = 64; // total width
uint8_t colour_depth = 5; // colour bit depth 1-6
uint8_t matrix_num = 1; // number of matrices
uint8_t row_addr_lines = 4; // matrix height inferred from this
bool double_buffer = true; // improve animation smoothness for extra ram

Adafruit_Protomatter matrix{matrix_width, colour_depth, matrix_num, rgbPins, 
                            row_addr_lines, addrPins, clockPin, latchPin,
                            oePin, double_buffer};

constexpr uint16_t hsv_red = 0;
constexpr uint16_t hsv_blue = 21840;
constexpr uint8_t center = 15;
constexpr uint8_t col0 = 4;
constexpr uint8_t col1 = 22;
constexpr uint8_t col2 = 46;
constexpr uint8_t col3 = 58;

StaticVec<StaticVec<NodePulsar, MAX_NODES>, NUM_LAYERS> node_matrix{{}, NUM_LAYERS};
StaticVec<StaticVec<StaticVec<LinkPulsar, MAX_LINKS>, MAX_NODES>, NUM_LAYERS> link_matrix{{}, NUM_LAYERS};
NodePulsar input1_node{col0, center-6, 2};
NodePulsar input2_node{col0, center+6, 2};


MLP mlp{};


void update_pulsar_brightnesses()
{
    MinMaxValues values = get_min_max_values(mlp);
    // input1 and 2 !!!
    for (int i{0}; i<link_matrix.size; ++i)
    {
        const StaticVec<Node, MAX_NODES>& nodes = mlp.get_layer(i).get_nodes();
        for (int j{0}; j<link_matrix[i].size; ++j)
        {
            const Node& node = nodes.arr[j];
            float output = node.get_output();
            float scaled = min_max_scale(output, values.node_min, values.node_max);
            float brightness = brightness_scale(scaled);
            node_matrix[i][j].set_max_brightness(brightness);
            for (int k{0}; k<link_matrix[i][j].size; ++k)
            {
                float link_strength = node.get_weights().arr[k] * node.get_inputs().arr[k];
                float scaled = min_max_scale(link_strength, values.link_min, values.link_max);
                float brightness = brightness_scale(scaled);
                link_matrix[i][j][k].set_max_brightness(brightness);
                // Serial.println(link_matrix[i][j][k].get_max_brightness());
            }
        }
    }
}

void setup() {

    Serial.begin(9600); // transmit to serial port @ 9600 bits per second

    ProtomatterStatus status = matrix.begin();
    Serial.print("Protomatter begin() status: ");
    Serial.println((int)status);
    if(status != PROTOMATTER_OK)
        for(;;);

    randomSeed(analogRead(0));
    mlp.init_weights();

    node_matrix[0][0] = NodePulsar{col1, center-12, 2};
    node_matrix[0][1] = NodePulsar{col1, center, 2};
    node_matrix[0][2] = NodePulsar{col1, center+12, 2};
    node_matrix[0].size = 3; 

    node_matrix[1][0] = NodePulsar{col2, center-12, 2};
    node_matrix[1][1] = NodePulsar{col2, center, 2};
    node_matrix[1][2] = NodePulsar{col2, center+12, 2};
    node_matrix[1].size = 3;

    node_matrix[2][0] = NodePulsar{col3, center, 2};
    node_matrix[2].size = 1;

    
    link_nodes(&input1_node, &node_matrix[0][0], &link_matrix[0][0][0]);
    link_nodes(&input1_node, &node_matrix[0][1], &link_matrix[0][0][1]);
    link_nodes(&input1_node, &node_matrix[0][2], &link_matrix[0][0][2]);
    link_matrix[0][0].size = 3;

    link_nodes(&input2_node, &node_matrix[0][0], &link_matrix[0][1][0]);
    link_nodes(&input2_node, &node_matrix[0][1], &link_matrix[0][1][1]);
    link_nodes(&input2_node, &node_matrix[0][2], &link_matrix[0][1][2]);
    link_matrix[0][1].size = 3;

    link_matrix[0].size = 2;

    link_nodes(&node_matrix[0][0], &node_matrix[1][0], &link_matrix[1][0][0]);
    link_nodes(&node_matrix[0][0], &node_matrix[1][1], &link_matrix[1][0][1]);
    link_nodes(&node_matrix[0][0], &node_matrix[1][2], &link_matrix[1][0][2]);
    link_matrix[1][0].size = 3;

    link_nodes(&node_matrix[0][1], &node_matrix[1][0], &link_matrix[1][1][0]);
    link_nodes(&node_matrix[0][1], &node_matrix[1][1], &link_matrix[1][1][1]);
    link_nodes(&node_matrix[0][1], &node_matrix[1][2], &link_matrix[1][1][2]);
    link_matrix[1][1].size = 3;
    
    link_nodes(&node_matrix[0][2], &node_matrix[1][0], &link_matrix[1][2][0]);
    link_nodes(&node_matrix[0][2], &node_matrix[1][1], &link_matrix[1][2][1]);
    link_nodes(&node_matrix[0][2], &node_matrix[1][2], &link_matrix[1][2][2]);
    link_matrix[1][2].size = 3;

    link_matrix[1].size = 3;

    link_nodes(&node_matrix[1][0], &node_matrix[2][0], &link_matrix[2][0][0]);
    link_matrix[2][0].size = 1;

    link_nodes(&node_matrix[1][1], &node_matrix[2][0], &link_matrix[2][1][0]);
    link_matrix[2][1].size = 1;

    link_nodes(&node_matrix[1][2], &node_matrix[2][0], &link_matrix[2][2][0]);
    link_matrix[2][2].size = 1;

    link_matrix[2].size = 3;
}

void loop() {

    static int instance = 0;
    static float cost = 0;
    static int i = 0;
    ++i;
    if (i == 1000)
    {
        i = 0;
        if (instance == TRAIN_DATA_SZ)
        {
            Serial.print("cost =");
            Serial.println(cost);
            cost = 0;
            instance = 0;
        }
        else
        {
            mlp.forward_pass(x_train.arr[instance], y_train[instance]);
            update_pulsar_brightnesses();
            mlp.backwards_pass(x_train.arr[instance], y_train[instance]);
            cost += mlp.get_cost();
            ++instance;
        }
        input1_node.init_pulse();
        input2_node.init_pulse();
    }

    for (int i{0}; i<link_matrix.size; ++i)
        for (int j{0}; j<link_matrix[i].size; ++j)
            for (int k{0}; k<link_matrix[i][j].size; ++k)
            {
                link_matrix[i][j][k].update();
                link_matrix[i][j][k].draw();
            }
    input1_node.update();
    input1_node.draw();
    input2_node.update();
    input2_node.draw();
    for (int i{0}; i<node_matrix.size; ++i)
        for (int j{0}; j<node_matrix[i].size; ++j)
        {
            node_matrix[i][j].update();
            node_matrix[i][j].draw();
        }
    matrix.show();


    // delay(1);
}
