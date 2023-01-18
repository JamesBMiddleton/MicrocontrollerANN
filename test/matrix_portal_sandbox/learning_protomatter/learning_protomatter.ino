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

}

void loop() 
{
  static uint16_t bright = 0;
  static int8_t step = 1;
  if (bright == 0)
    step = 1;
  if (bright == 120)
    step = -1;

  uint16_t c = matrix.color565(bright, bright, bright);

  constexpr int hsv_convert = 182;

  bright += step;

  uint8_t sat = abs(bright - 60);

  //matrix.writeLine(0, bright, 64, bright, matrix.color565(bright*8, 255-bright*8, 0));

  constexpr uint8_t center = 15;

  constexpr uint8_t col0 = 4;
  constexpr uint8_t col1 = 22;
  constexpr uint8_t col2 = 46;
  constexpr uint8_t col3 = 58;

  Serial.println(sat);
  
  matrix.writeLine(col0, center-6, col1, center-12, matrix.colorHSV(bright*hsv_convert, 192 + sat, bright));
  matrix.writeLine(col0, center-6, col1, center, matrix.colorHSV(120*hsv_convert, 255, 255));
  matrix.writeLine(col0, center-6, col1, center+12, matrix.colorHSV(240*hsv_convert, 255, 255));

  matrix.writeLine(col0, center+6, col1, center-12, matrix.color565(128, 64, 64));
  matrix.writeLine(col0, center+6, col1, center, luminance(bright));
  matrix.writeLine(col0, center+6, col1, center+12, luminance(bright));


  matrix.writeLine(col1, center-12, col2, center-12, matrix.color565(bright, 255-bright, 50));
  matrix.writeLine(col1, center-12, col2, center, matrix.color565(255-bright, bright, 50));
  matrix.writeLine(col1, center-12, col2, center+12, matrix.color565(255, 50, 20));

  matrix.writeLine(col1, center, col2, center-12, matrix.color565(255-bright, bright, 50));
  matrix.writeLine(col1, center, col2, center, matrix.color565(128, 128, 20));
  matrix.writeLine(col1, center, col2, center+12, matrix.color565(bright, 255-bright, 50));
  
  matrix.writeLine(col1, center+12, col2, center-12, matrix.color565(8, 8, 8));
  matrix.writeLine(col1, center+12, col2, center, matrix.color565(128, 128, 128));
  matrix.writeLine(col1, center+12, col2, center+12, matrix.color565(50, 50, 50));


  matrix.writeLine(col2, center-12, col3, center, matrix.color565(8, 8, 8));
  matrix.writeLine(col2, center, col3, center, matrix.color565(128, 128, 128));
  matrix.writeLine(col2, center+12, col3, center, matrix.color565(50, 50, 50));

 
  matrix.fillCircle(col0, center-6, 2, matrix.color565(8, 8, 8));
  matrix.fillCircle(col0, center+6, 2, matrix.color565(128, 128, 128));

  matrix.fillCircle(col1, center-12, 2, matrix.color565(32, 32, 32));
  matrix.fillCircle(col1, center, 2, matrix.color565(200, 200, 200));
  matrix.fillCircle(col1, center+12, 2, matrix.color565(64, 64, 64));

  matrix.fillCircle(col2, center-12, 2, matrix.color565(100, 100, 100));
  matrix.fillCircle(col2, center, 2, matrix.color565(150, 150, 150));
  matrix.fillCircle(col2, center+12, 2, matrix.color565(16, 16, 16));

  matrix.fillCircle(col3, center, 2, matrix.color565(32, 32, 32));
  
  matrix.show();
  delay(100);
}
