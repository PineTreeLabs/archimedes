#include <Wire.h>


static inline int arduino_i2c_write(void *handle, uint8_t dev, uint8_t reg,
  uint8_t *data, uint16_t len) {
  Wire.beginTransmission(dev);
  Wire.write(reg); 
  for (uint16_t i = 0; i < len; i++) {
    Wire.write(data[i]);
  }
  return (Wire.endTransmission() == 0) ? 0 : -1;
}

static inline int arduino_i2c_read(void *handle, uint8_t dev, uint8_t reg,
                     uint8_t *data, uint16_t len) {
    Wire.beginTransmission(dev);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return -1;
    Wire.requestFrom(dev, len);
    for (uint16_t i = 0; i < len; i++) {
        if (!Wire.available()) return -1;
        data[i] = Wire.read();
    }
    return 0;
}