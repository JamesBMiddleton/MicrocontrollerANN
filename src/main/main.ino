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

// constexpr uint8_t num_nodes = 9;
// constexpr uint8_t num_links = 18;

StaticVec<StaticVec<NodePulsar, MAX_NODES>, NUM_LAYERS> node_matrix;//{{}, NUM_LAYERS};
StaticVec<StaticVec<StaticVec<LinkPulsar, MAX_LINKS>, MAX_NODES>, NUM_LAYERS> link_matrix;//{{}, NUM_LAYERS};
NodePulsar input1_node{col0, center-6, 2};
NodePulsar input2_node{col0, center+6, 2};


MLP mlp{};


void setup() {

    Serial.begin(9600); // transmit to serial port @ 9600 bits per second

    ProtomatterStatus status = matrix.begin();
    Serial.print("Protomatter begin() status: ");
    Serial.println((int)status);
    if(status != PROTOMATTER_OK)
        for(;;);

    node_matrix[0][0] = NodePulsar{col1, center-12, 2};
    node_matrix[0][1] = NodePulsar{col1, center, 2};
    node_matrix[0][2] = NodePulsar{col1, center+12, 2};
    node_matrix[0].size = MAX_NODES;

    node_matrix[1][0] = NodePulsar{col2, center-12, 2};
    node_matrix[1][1] = NodePulsar{col2, center, 2};
    node_matrix[1][2] = NodePulsar{col2, center+12, 2};
    node_matrix[1].size = MAX_NODES;

    node_matrix[2][0] = NodePulsar{col3, center, 2};
    node_matrix[2].size = 1;

    
    // link_nodes(&input1_node, &node_matrix.arr[0].arr[0], &link_matrix.arr[0].);
    // link_nodes(&node_arr[0], &node_arr[3], &link_arr[1]);
    // link_nodes(&node_arr[0], &node_arr[4], &link_arr[2]);
    //
    // link_nodes(&node_arr[1], &node_arr[2], &link_arr[3]);
    // link_nodes(&node_arr[1], &node_arr[3], &link_arr[4]);
    // link_nodes(&node_arr[1], &node_arr[4], &link_arr[5]);
    //
    //
    // link_nodes(&node_arr[2], &node_arr[5], &link_arr[6]);
    // link_nodes(&node_arr[2], &node_arr[6], &link_arr[7]);
    // link_nodes(&node_arr[2], &node_arr[7], &link_arr[8]);
    //
    // link_nodes(&node_arr[3], &node_arr[5], &link_arr[9]);
    // link_nodes(&node_arr[3], &node_arr[6], &link_arr[10]);
    // link_nodes(&node_arr[3], &node_arr[7], &link_arr[11]);
    //
    // link_nodes(&node_arr[4], &node_arr[5], &link_arr[12]);
    // link_nodes(&node_arr[4], &node_arr[6], &link_arr[13]);
    // link_nodes(&node_arr[4], &node_arr[7], &link_arr[14]);
    //
    // link_nodes(&node_arr[5], &node_arr[8], &link_arr[15]);
    // link_nodes(&node_arr[6], &node_arr[8], &link_arr[16]);
    // link_nodes(&node_arr[7], &node_arr[8], &link_arr[17]);
    //
    // node_arr[0].set_max_brightness(9);
    // node_arr[3].set_max_brightness(9);
    // node_arr[5].set_max_brightness(255);
    //
    // link_arr[0].set_max_brightness(9);
    // link_arr[1].set_max_brightness(9);
    // link_arr[2].set_max_brightness(9);
    //
    // link_arr[4].set_max_brightness(9);
    //
    // link_arr[6].set_max_brightness(128);
    // link_arr[7].set_max_brightness(9);
    // link_arr[8].set_max_brightness(9);
    //
    // link_arr[9].set_max_brightness(9);
    // link_arr[10].set_max_brightness(9);
    // link_arr[11].set_max_brightness(9);
    // link_arr[12].set_max_brightness(9);
    //
    // node_arr[0].init_pulse();
    // node_arr[1].init_pulse();
    //
    randomSeed(analogRead(0));
    mlp.init_weights();
}

void loop() {

    static float lowest_cost = 10000;
    float cost = 0;
    for (int j{0}; j < TRAIN_DATA_SZ; ++j)  
    {
        mlp.forward_pass(x_train.arr[j], y_train[j]);

        MinMaxValues v = get_min_max_values(mlp);

        mlp.backwards_pass(x_train.arr[j], y_train[j]);
        cost += mlp.get_cost();
        // delay(1);
    }
    Serial.println(cost);
    if (cost < lowest_cost)
        lowest_cost = cost;

    // Serial.println(lowest_cost);
    //
    // for (int i{0}; i<num_links; ++i)
    // {
    //     link_arr[i].update();
    //     link_arr[i].draw();
    // }
    // for (int i{0}; i<num_nodes; ++i)
    // {
    //     node_arr[i].update();
    //     node_arr[i].draw();
    // }
    // matrix.show();
    // // Serial.println(node_arr[2].get_brightness());
    // 
    // static int i = 0;
    // ++i;
    // if (i == 1000)
    // {
    //     node_arr[0].init_pulse();
    //     node_arr[1].init_pulse();
    //     i = 0;
    // }

    delay(100);
}
