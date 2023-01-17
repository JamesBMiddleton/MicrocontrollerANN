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
  static uint8_t bright = 0;
  static uint8_t step = 1;
  if (bright == 0)
    step = 1;
  if (bright == 150)
    step = -1;

  matrix.fillScreen(0);
  //matrix.fillCircle(32, 16, 10, matrix.color565(bright, bright, bright));

  matrix.setCursor(16, 8);
  matrix.println("DON'T");
  matrix.setCursor(16, 16);
  matrix.println("PANIC!");
  matrix.setTextColor(matrix.color565(bright, bright, bright));
  bright += step;
  
  matrix.show();
  delay(1);
}
