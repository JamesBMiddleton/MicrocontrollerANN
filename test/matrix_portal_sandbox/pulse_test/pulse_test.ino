#include <Adafruit_Protomatter.h>
#include "pulsar.h"

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

constexpr uint16_t hsv_red = 0;
constexpr uint16_t hsv_blue = 21840;
constexpr uint8_t center = 15;
constexpr uint8_t col0 = 4;
constexpr uint8_t col1 = 22;
constexpr uint8_t col2 = 46;
constexpr uint8_t col3 = 58;

constexpr uint8_t num_nodes = 9;
constexpr uint8_t num_links = 18;

Adafruit_Protomatter matrix{matrix_width, colour_depth, matrix_num, rgbPins, 
                            row_addr_lines, addrPins, clockPin, latchPin,
                            oePin, double_buffer};

NodePulsar node_arr[num_nodes];
LinkPulsar link_arr[num_links];

void setup() {
    Serial.begin(9600); // transmit to serial port @ 9600 bits per second

    ProtomatterStatus status = matrix.begin();
    Serial.print("Protomatter begin() status: ");
    Serial.println((int)status);
    if(status != PROTOMATTER_OK)
        for(;;);

    // node_arr[0] = NodePulsar{32, 16, 2};
    // node_arr[1] = NodePulsar{45, 8, 2};
    // link_arr[0] = link_nodes(&node_arr[0], &node_arr[1]);
    node_arr[0] = NodePulsar{col0, center-6, 2};
    node_arr[1] = NodePulsar{col0, center+6, 2};
    
    node_arr[2] = NodePulsar{col1, center-12, 2};
    node_arr[3] = NodePulsar{col1, center, 2};
    node_arr[4] = NodePulsar{col1, center+12, 2};

    node_arr[5] = NodePulsar{col2, center-12, 2};
    node_arr[6] = NodePulsar{col2, center, 2};
    node_arr[7] = NodePulsar{col2, center+12, 2};

    node_arr[8] = NodePulsar{col3, center, 2};

    link_nodes(&node_arr[0], &node_arr[2], &link_arr[0]);
    link_nodes(&node_arr[0], &node_arr[3], &link_arr[1]);
    link_nodes(&node_arr[0], &node_arr[4], &link_arr[2]);

    link_nodes(&node_arr[1], &node_arr[2], &link_arr[3]);
    link_nodes(&node_arr[1], &node_arr[3], &link_arr[4]);
    link_nodes(&node_arr[1], &node_arr[4], &link_arr[5]);


    link_nodes(&node_arr[2], &node_arr[5], &link_arr[6]);
    link_nodes(&node_arr[2], &node_arr[6], &link_arr[7]);
    link_nodes(&node_arr[2], &node_arr[7], &link_arr[8]);
    
    link_nodes(&node_arr[3], &node_arr[5], &link_arr[9]);
    link_nodes(&node_arr[3], &node_arr[6], &link_arr[10]);
    link_nodes(&node_arr[3], &node_arr[7], &link_arr[11]);

    link_nodes(&node_arr[4], &node_arr[5], &link_arr[12]);
    link_nodes(&node_arr[4], &node_arr[6], &link_arr[13]);
    link_nodes(&node_arr[4], &node_arr[7], &link_arr[14]);

    link_nodes(&node_arr[5], &node_arr[8], &link_arr[15]);
    link_nodes(&node_arr[6], &node_arr[8], &link_arr[16]);
    link_nodes(&node_arr[7], &node_arr[8], &link_arr[17]);

    node_arr[0].init_pulse();
    node_arr[1].init_pulse();

}

void loop() {
    for (int i{0}; i<num_links; ++i)
    {
        link_arr[i].update();
        link_arr[i].draw();
    }
    for (int i{0}; i<num_nodes; ++i)
    {
        node_arr[i].update();
        node_arr[i].draw();
    }
    matrix.show();
    delay(1);
}
