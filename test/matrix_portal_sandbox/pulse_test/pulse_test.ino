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

Adafruit_Protomatter matrix{matrix_width, colour_depth, matrix_num, rgbPins, 
                            row_addr_lines, addrPins, clockPin, latchPin,
                            oePin, double_buffer};

LinkPulsarArray array2{{}, 0};
NodePulsar node2{45, 8, 2, array2};
LinkPulsar link2{32, 16, 45, 8, &node2};


LinkPulsarArray array{{&link2}, 1};
NodePulsar node{32, 16, 2, array};
LinkPulsar link{16, 8, 32, 16, &node};


constexpr uint16_t hsv_red = 0;
constexpr uint16_t hsv_blue = 21840;
constexpr uint8_t center = 15;
constexpr uint8_t col0 = 4;
constexpr uint8_t col1 = 22;
constexpr uint8_t col2 = 46;
constexpr uint8_t col3 = 58;

void setup() {
    Serial.begin(9600); // transmit to serial port @ 9600 bits per second

    ProtomatterStatus status = matrix.begin();
    Serial.print("Protomatter begin() status: ");
    Serial.println((int)status);
    if(status != PROTOMATTER_OK)
        for(;;);

    link.init_pulse();

}

void loop() {
    link.update();
    link2.update();
    node.update();
    node2.update();

    link.draw();
    link2.draw();
    node.draw();
    node2.draw();
    
    matrix.show();
    delay(1);
}
