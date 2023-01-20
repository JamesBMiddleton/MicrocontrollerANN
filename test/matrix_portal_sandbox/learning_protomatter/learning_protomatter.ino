#include <Adafruit_Protomatter.h>

uint8_t rgbPins[]  = {7, 8, 9, 10, 11, 12};
uint8_t addrPins[] = {17, 18, 19, 20};
uint8_t clockPin   = 14;
uint8_t latchPin   = 15;
uint8_t oePin      = 16;

uint8_t matrix_width = 64; // total width
uint8_t colour_depth = 5; // colour bit depth 1-6
uint8_t matrix_num = 1; // number of matrices
uint8_t row_addr_lines = 4; // matrix height inferred from this
uint8_t double_buffer = true; // improve animation smoothness for extra ram

constexpr uint16_t hsv_red = 0;
constexpr uint16_t hsv_blue = 21840;
constexpr uint8_t center = 15;
constexpr uint8_t col0 = 4;
constexpr uint8_t col1 = 22;
constexpr uint8_t col2 = 46;
constexpr uint8_t col3 = 58;


Adafruit_Protomatter matrix{matrix_width, colour_depth, matrix_num, rgbPins, 
                            row_addr_lines, addrPins, clockPin, latchPin,
                            oePin, double_buffer};

uint16_t luminance(uint8_t l)
{
  return matrix.color565(l, l, l);
}

void setup()
{
    Serial.begin(9600); // transmit to serial port @ 9600 bits per second

    ProtomatterStatus status = matrix.begin();
    Serial.print("Protomatter begin() status: ");
    Serial.println((int)status);
    if(status != PROTOMATTER_OK)
        for(;;);

    matrix.writeLine(col0, center-6, col1, center-12, matrix.colorHSV(hsv_red, 255, 8));
    matrix.writeLine(col0, center-6, col1, center, matrix.colorHSV(hsv_blue, 255, 8));
    matrix.writeLine(col0, center-6, col1, center+12, matrix.colorHSV(hsv_blue, 255, 8));

    matrix.writeLine(col0, center+6, col1, center-12, matrix.colorHSV(hsv_blue, 255, 8));
    matrix.writeLine(col0, center+6, col1, center, matrix.colorHSV(hsv_red, 255, 8));
    matrix.writeLine(col0, center+6, col1, center+12, matrix.colorHSV(hsv_red, 255, 8));


    matrix.writeLine(col1, center-12, col2, center-12, matrix.colorHSV(hsv_red, 255, 8));
    matrix.writeLine(col1, center-12, col2, center, matrix.colorHSV(hsv_blue, 255, 8));
    matrix.writeLine(col1, center-12, col2, center+12, matrix.colorHSV(hsv_blue, 255, 8));

    matrix.writeLine(col1, center, col2, center-12, matrix.colorHSV(hsv_blue, 255, 8));
    matrix.writeLine(col1, center, col2, center, matrix.colorHSV(hsv_red, 255, 8));
    matrix.writeLine(col1, center, col2, center+12, matrix.colorHSV(hsv_red, 255, 8));

    matrix.writeLine(col1, center+12, col2, center-12, matrix.colorHSV(hsv_blue, 255, 8));
    matrix.writeLine(col1, center+12, col2, center, matrix.colorHSV(hsv_red, 255, 8));
    matrix.writeLine(col1, center+12, col2, center+12, matrix.colorHSV(hsv_blue, 255, 8));


    matrix.writeLine(col2, center-12, col3, center, matrix.colorHSV(hsv_red, 255, 8));
    matrix.writeLine(col2, center, col3, center, matrix.colorHSV(hsv_blue, 255, 8));
    matrix.writeLine(col2, center+12, col3, center, matrix.colorHSV(hsv_red, 255, 8));


    matrix.fillCircle(col0, center-6, 2, matrix.colorHSV(0, 0, 8));
    matrix.fillCircle(col0, center+6, 2, matrix.colorHSV(0, 0, 8));

    matrix.fillCircle(col1, center-12, 2, matrix.colorHSV(0, 0, 8));
    matrix.fillCircle(col1, center, 2, matrix.colorHSV(0, 0, 8));
    matrix.fillCircle(col1, center+12, 2, matrix.colorHSV(0, 0, 8));

    matrix.fillCircle(col2, center-12, 2, matrix.colorHSV(0, 0, 8));
    matrix.fillCircle(col2, center, 2, matrix.colorHSV(0, 0, 8));
    matrix.fillCircle(col2, center+12, 2, matrix.colorHSV(0, 0, 8));

    matrix.fillCircle(col3, center, 2, matrix.colorHSV(0, 0, 8));

    matrix.show();
}

void loop() 
{
    static uint16_t bright = 8;
    static int8_t step = 1;
    if (bright == 8)
        step = 1;
    if (bright == 255)
        step = -1;
    bright += step;

    int16_t lag = bright - 128;
    uint16_t bright2 = (lag > 0) ? bright - 120 : 8;
    Serial.println(bright);
    Serial.println(lag);

    matrix.writeLine(col0, center-6, col1, center-12, matrix.colorHSV(0, 255, bright2));
    matrix.fillCircle(col0, center-6, 2, matrix.colorHSV(0, 0, bright));

    matrix.show();

    int* i = new int{1};
    delete i;

    delay(1);
}
