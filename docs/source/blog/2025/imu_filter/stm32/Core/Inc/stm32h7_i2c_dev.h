#ifndef STM32_I2C_DEV_H
#define STM32_I2C_DEV_H

#include "stm32h7xx_hal.h"

/* STM32 I2C wrapper functions
 * Pass I2C handle as the 'handle' parameter
 */

typedef struct {
    I2C_HandleTypeDef *hi2c;
    uint8_t dev_addr;
} stm32_i2c_handle_t;

static inline int stm32_i2c_write(
    void *handle, uint8_t reg, uint8_t *data, uint16_t len) {
    stm32_i2c_handle_t *i2c_handle = (stm32_i2c_handle_t *)handle;
    
    // Shift device address to 7-bit format (STM32 HAL expects 7-bit << 1)
    uint16_t dev_addr = i2c_handle->dev_addr << 1;
    
    // Write register address followed by data
    HAL_StatusTypeDef status = HAL_I2C_Mem_Write(
        i2c_handle->hi2c, 
        dev_addr, 
        reg, 
        I2C_MEMADD_SIZE_8BIT,
        data, 
        len, 
        HAL_MAX_DELAY
    );
    
    return (status == HAL_OK) ? 0 : -1;
}

static inline int stm32_i2c_read(
    void *handle, uint8_t reg, uint8_t *data, uint16_t len) {
    stm32_i2c_handle_t *i2c_handle = (stm32_i2c_handle_t *)handle;
    
    // Shift device address to 7-bit format
    uint16_t dev_addr = i2c_handle->dev_addr << 1;
    
    // Read from register
    HAL_StatusTypeDef status = HAL_I2C_Mem_Read(
        i2c_handle->hi2c,
        dev_addr,
        reg,
        I2C_MEMADD_SIZE_8BIT,
        data,
        len,
        HAL_MAX_DELAY
    );
    
    return (status == HAL_OK) ? 0 : -1;
}

#endif // STM32_I2C_DEV_H