#ifndef STM32_SPI_DEV_H
#define STM32_SPI_DEV_H

#include "stm32h7xx_hal.h"

#define STM32_SPI_MAX_TRANSFER_SIZE 256

/* STM32 SPI wrapper functions
 * Pass SPI handle and CS GPIO info via the handle structure
 */

typedef struct
{
    SPI_HandleTypeDef *hspi;
    GPIO_TypeDef *cs_port;
    uint16_t cs_pin;
    uint8_t tx_buffer[STM32_SPI_MAX_TRANSFER_SIZE];
    uint8_t rx_buffer[STM32_SPI_MAX_TRANSFER_SIZE];
} stm32_spi_handle_t;

static inline void stm32_spi_init(stm32_spi_handle_t *handle)
{
    memset(handle->tx_buffer, 0, sizeof(handle->tx_buffer));
    memset(handle->rx_buffer, 0, sizeof(handle->rx_buffer));

    // CS high initially (deselected)
    HAL_GPIO_WritePin(handle->cs_port, handle->cs_pin, GPIO_PIN_SET);
}

static inline int stm32_spi_write(
    void *handle, uint8_t reg, uint8_t *data, uint16_t len)
{
    stm32_spi_handle_t *spi_handle = (stm32_spi_handle_t *)handle;
    HAL_StatusTypeDef status;

    // Allocate buffer for register address + data
    if (len + 1 > sizeof(spi_handle->tx_buffer))
    {
        return -1; // Transfer too large
    }

    // First byte is register address with write bit (bit 7 = 0)
    spi_handle->tx_buffer[0] = reg & 0x7F;

    // Copy data to transmit buffer
    for (uint16_t i = 0; i < len; i++)
    {
        spi_handle->tx_buffer[i + 1] = data[i];
    }

    // Assert CS (active low)
    HAL_GPIO_WritePin(spi_handle->cs_port, spi_handle->cs_pin, GPIO_PIN_RESET);

    // Transmit register address + data
    status = HAL_SPI_Transmit(spi_handle->hspi, spi_handle->tx_buffer, len + 1, HAL_MAX_DELAY);

    // Deassert CS
    HAL_GPIO_WritePin(spi_handle->cs_port, spi_handle->cs_pin, GPIO_PIN_SET);

    return (status == HAL_OK) ? 0 : -1;
}

static inline int stm32_spi_read(
    void *handle, uint8_t reg, uint8_t *data, uint16_t len)
{
    stm32_spi_handle_t *spi_handle = (stm32_spi_handle_t *)handle;
    HAL_StatusTypeDef status;

    // Allocate buffers for transmit and receive
    if (len + 1 > sizeof(spi_handle->tx_buffer))
    {
        return -1; // Transfer too large
    }

    // First byte is register address with read bit (bit 7 = 1)
    spi_handle->tx_buffer[0] = reg | 0x80;

    // Fill rest of tx_buffer with dummy bytes (0x00)
    for (uint16_t i = 1; i <= len; i++)
    {
        spi_handle->tx_buffer[i] = 0x00;
    }

    // Assert CS (active low)
    HAL_GPIO_WritePin(spi_handle->cs_port, spi_handle->cs_pin, GPIO_PIN_RESET);

    // Transmit register address + dummy bytes, receive response
    status = HAL_SPI_TransmitReceive(
        spi_handle->hspi, spi_handle->tx_buffer, spi_handle->rx_buffer,
        len + 1, HAL_MAX_DELAY);

    // Deassert CS
    HAL_GPIO_WritePin(spi_handle->cs_port, spi_handle->cs_pin, GPIO_PIN_SET);

    if (status != HAL_OK)
    {
        return -1;
    }

    // Copy received data (skip first byte which is the response to address)
    for (uint16_t i = 0; i < len; i++)
    {
        data[i] = spi_handle->rx_buffer[i + 1];
    }

    return 0;
}

#endif // STM32_SPI_DEV_H