#include <stdint.h>
#include <stdbool.h>

#define LSM6DSOX_I2CADDR_DEFAULT 0x6A ///< LSM6DS default i2c address

#define LSM6DSOX_FUNC_CFG_ACCESS 0x1 ///< Enable embedded functions register
#define LSM6DSOX_INT1_CTRL 0x0D      ///< Interrupt control for INT 1
#define LSM6DSOX_INT2_CTRL 0x0E      ///< Interrupt control for INT 2
#define LSM6DSOX_WHOAMI 0x0F         ///< Chip ID register
#define LSM6DSOX_CTRL1_XL 0x10       ///< Main accelerometer config register
#define LSM6DSOX_CTRL2_G 0x11        ///< Main gyro config register
#define LSM6DSOX_CTRL3_C 0x12        ///< Main configuration register
#define LSM6DSOX_CTRL8_XL 0x17       ///< High and low pass for accel
#define LSM6DSOX_CTRL10_C 0x19       ///< Main configuration register
#define LSM6DSOX_WAKEUP_SRC 0x1B     ///< Why we woke up
#define LSM6DSOX_STATUS_REG 0X1E     ///< Status register
#define LSM6DSOX_OUT_TEMP_L 0x20     ///< First data register (temperature low)
#define LSM6DSOX_OUTX_L_G 0x22       ///< First gyro data register
#define LSM6DSOX_OUTX_L_A 0x28       ///< First accel data register
#define LSM6DSOX_STEPCOUNTER 0x4B    ///< 16-bit step counter
#define LSM6DSOX_TAP_CFG 0x58        ///< Tap/pedometer configuration
#define LSM6DSOX_WAKEUP_THS \
  0x5B ///< Single and double-tap function threshold register
#define LSM6DSOX_WAKEUP_DUR \
  0x5C                        ///< Free-fall, wakeup, timestamp and sleep mode duration
#define LSM6DSOX_MD1_CFG 0x5E ///< Functions routing on INT1 register

#define LSM6DSOX_OUTPUT_RES 32768 // 16-bit output resolution (+ sign bit)

// Bit position definitions
#define ODR_POS 4 // ODR starts at bit 4
#define ODR_MASK (0x0F << ODR_POS)
#define MAKE_ODR(val) (((val) << ODR_POS) & ODR_MASK)

#define ACCEL_FS_POS 2 // Full-scale at bit 2
#define ACCEL_FS_MASK (0x03 << ACCEL_FS_POS)
#define MAKE_ACCEL_FS(val) (((val) << ACCEL_FS_POS) & ACCEL_FS_MASK)

#define GYRO_FS_POS 1                      // Gyro full-scale at bit 1
#define GYRO_FS_MASK (0x07 << GYRO_FS_POS) // 3 bits for gyro
#define MAKE_GYRO_FS(val) (((val) << GYRO_FS_POS) & GYRO_FS_MASK)

#define CTRL1_LPF2_XL_EN_POS 1 // LPF2 enable at bit 1
#define CTRL1_LPF2_XL_EN_MASK (0x01 << CTRL1_LPF2_XL_EN_POS)

#define CTRL8_HPCF_XL_POS 5 // Filter cutoff starts at bit 5
#define CTRL8_HPCF_XL_MASK (0x07 << CTRL8_HPCF_XL_POS)
#define CTRL8_HP_SLOPE_XL_EN_POS 2 // HPF enable at bit 2
#define CTRL8_HP_SLOPE_XL_EN_MASK (0x01 << CTRL8_HP_SLOPE_XL_EN_POS)
#define CTRL8_HP_REF_MODE (0 << 4)     // No HPF reference mode
#define CTRL8_FAST_SETTL_MODE (0 << 3) // Normal mode
#define CTRL8_XL_XL_FS_MODE (0 << 1)
#define CTRL8_XL_LOW_PASS_6D (0 << 0)

// Bit field macros
#define MAKE_LPF2_XL_EN(val) (((val) << CTRL1_LPF2_XL_EN_POS) & CTRL1_LPF2_XL_EN_MASK)
#define MAKE_HPCF_XL(val) (((val) << CTRL8_HPCF_XL_POS) & CTRL8_HPCF_XL_MASK)
#define MAKE_HP_SLOPE_XL_EN(val) (((val) << CTRL8_HP_SLOPE_XL_EN_POS) & CTRL8_HP_SLOPE_XL_EN_MASK)

#ifndef LSM6DSOX_H
#define LSM6DSOX_H

/** output data rates */
typedef enum data_rate
{
  LSM6DSOX_RATE_SHUTDOWN = 0b0000,
  LSM6DSOX_RATE_12_5_HZ = 0b0001,
  LSM6DSOX_RATE_26_HZ = 0b0010,
  LSM6DSOX_RATE_52_HZ = 0b0011,
  LSM6DSOX_RATE_104_HZ = 0b0100,
  LSM6DSOX_RATE_208_HZ = 0b0101,
  LSM6DSOX_RATE_416_HZ = 0b0110,
  LSM6DSOX_RATE_833_HZ = 0b0111,
  LSM6DSOX_RATE_1_66K_HZ = 0b1000,
  LSM6DSOX_RATE_3_33K_HZ = 0b1001,
  LSM6DSOX_RATE_6_66K_HZ = 0b1010,
} lsm6dsox_data_rate_t;

/* accelerometer data range */
// NOTE: Assumes XL_FS_MODE = 0 (no OIS) in CTRL8_XL
// otherwise 0b01 maps to ±2g
typedef enum accel_range
{
  LSM6DSOX_ACCEL_RANGE_2_G = 0b00,
  LSM6DSOX_ACCEL_RANGE_16_G = 0b01,
  LSM6DSOX_ACCEL_RANGE_4_G = 0b10,
  LSM6DSOX_ACCEL_RANGE_8_G = 0b11
} lsm6dsox_accel_range_t;

/** gyro data range */
typedef enum gyro_range
{
  LSM6DSOX_GYRO_RANGE_125_DPS = 0b001,
  LSM6DSOX_GYRO_RANGE_250_DPS = 0b000,
  LSM6DSOX_GYRO_RANGE_500_DPS = 0b010,
  LSM6DSOX_GYRO_RANGE_1000_DPS = 0b100,
  LSM6DSOX_GYRO_RANGE_2000_DPS = 0b110,
} lsm6dsox_gyro_range_t;

/** filter bandwidth (9.22 in datasheet) */
typedef enum filter_cutoff
{
  LSM6DSOX_HPF_ODR_DIV_4 = 0b000,
  LSM6DSOX_HPF_ODR_DIV_10 = 0b001,
  LSM6DSOX_HPF_ODR_DIV_20 = 0b010,
  LSM6DSOX_HPF_ODR_DIV_45 = 0b011,
  LSM6DSOX_HPF_ODR_DIV_100 = 0b100,
  LSM6DSOX_HPF_ODR_DIV_200 = 0b101,
  LSM6DSOX_HPF_ODR_DIV_400 = 0b110,
  LSM6DSOX_HPF_ODR_DIV_800 = 0b111
} lsm6dsox_filter_cutoff_t;

// Note: skip filters for now (just use defaults)
typedef struct
{
  // Hardware interface
  void *handle;

  lsm6dsox_data_rate_t accel_odr;
  lsm6dsox_accel_range_t accel_range;
  lsm6dsox_data_rate_t gyro_odr;
  lsm6dsox_gyro_range_t gyro_range;

  // State
  bool initialized;

  // Calibration
  float accel_bias[3];
  float gyro_bias[3];

  // Function pointers for HAL
  int (*read)(void *handle, uint8_t reg, uint8_t *data, uint16_t len);
  int (*write)(void *handle, uint8_t reg, uint8_t *data, uint16_t len);
} lsm6dsox_dev_t;

typedef struct
{
  uint8_t buffer[14];
  float temp;
  float gyro[3];
  float accel[3];
} lsm6dsox_data_t;

/* Create CTRL1_XL register byte */
static inline uint8_t lsm6dsox_get_ctrl1(
    lsm6dsox_data_rate_t odr, lsm6dsox_accel_range_t full_scale, bool LPF2_XL_EN)
{
  return MAKE_ODR(odr) | MAKE_ACCEL_FS(full_scale) | MAKE_LPF2_XL_EN(LPF2_XL_EN);
}

/* Create CTRL2_G register byte */
static inline uint8_t lsm6dsox_get_ctrl2(
    lsm6dsox_data_rate_t odr, lsm6dsox_gyro_range_t full_scale)
{
  return MAKE_ODR(odr) | MAKE_GYRO_FS(full_scale);
}

static inline uint8_t lsm6dsox_get_ctrl8(
    bool HP_SLOPE_XL_EN, lsm6dsox_filter_cutoff_t HPCF_XL)
{
  uint8_t ctrl8 = 0;
  ctrl8 |= MAKE_HPCF_XL(HPCF_XL);               // HPCF_XL: shift to 7-5
  ctrl8 |= CTRL8_HP_REF_MODE;                   // bit 4
  ctrl8 |= CTRL8_FAST_SETTL_MODE;               // bit 3
  ctrl8 |= MAKE_HP_SLOPE_XL_EN(HP_SLOPE_XL_EN); // bit 2
  ctrl8 |= CTRL8_XL_XL_FS_MODE;                 // bit 1
  ctrl8 |= CTRL8_XL_LOW_PASS_6D;
  return ctrl8;
}

/* Output data rate in Hz */
static inline float lsm6dsox_get_data_rate(lsm6dsox_data_rate_t rate)
{
  switch (rate)
  {
  case LSM6DSOX_RATE_SHUTDOWN:
    return 0.0;
  case LSM6DSOX_RATE_12_5_HZ:
    return 12.5;
  case LSM6DSOX_RATE_26_HZ:
    return 26.0;
  case LSM6DSOX_RATE_52_HZ:
    return 52.0;
  case LSM6DSOX_RATE_104_HZ:
    return 104.0;
  case LSM6DSOX_RATE_208_HZ:
    return 208.0;
  case LSM6DSOX_RATE_416_HZ:
    return 416.0;
  case LSM6DSOX_RATE_833_HZ:
    return 833.0;
  case LSM6DSOX_RATE_1_66K_HZ:
    return 1660.0;
  case LSM6DSOX_RATE_3_33K_HZ:
    return 3330.0;
  case LSM6DSOX_RATE_6_66K_HZ:
    return 6660.0;
  default:
    return 0.0;
  }
}

/* Accelerometer range in mg's */
static inline float lsm6dsox_get_accel_range(lsm6dsox_accel_range_t range)
{
  switch (range)
  {
  case LSM6DSOX_ACCEL_RANGE_2_G:
    return 2.0;
  case LSM6DSOX_ACCEL_RANGE_4_G:
    return 4.0;
  case LSM6DSOX_ACCEL_RANGE_8_G:
    return 8.0;
  case LSM6DSOX_ACCEL_RANGE_16_G:
    return 16.0;
  default:
    return 0.0;
  }
}

/* Gyroscope range in dps */
static inline float lsm6dsox_get_gyro_range(lsm6dsox_gyro_range_t range)
{
  switch (range)
  {
  case LSM6DSOX_GYRO_RANGE_125_DPS:
    return 125.0;
  case LSM6DSOX_GYRO_RANGE_250_DPS:
    return 250.0;
  case LSM6DSOX_GYRO_RANGE_500_DPS:
    return 500.0;
  case LSM6DSOX_GYRO_RANGE_1000_DPS:
    return 1000.0;
  case LSM6DSOX_GYRO_RANGE_2000_DPS:
    return 2000.0;
  default:
    return 0.0;
  }
}

/* Filter cutoff frequency: ODR divided by return */
static inline uint16_t lsm6dsox_get_filter_cutoff(lsm6dsox_filter_cutoff_t cutoff)
{
  switch (cutoff)
  {
  case LSM6DSOX_HPF_ODR_DIV_4:
    return 4;
  case LSM6DSOX_HPF_ODR_DIV_10:
    return 10;
  case LSM6DSOX_HPF_ODR_DIV_20:
    return 20;
  case LSM6DSOX_HPF_ODR_DIV_45:
    return 45;
  case LSM6DSOX_HPF_ODR_DIV_100:
    return 100;
  case LSM6DSOX_HPF_ODR_DIV_200:
    return 200;
  case LSM6DSOX_HPF_ODR_DIV_400:
    return 400;
  case LSM6DSOX_HPF_ODR_DIV_800:
    return 800;
  default:
    return 0;
  }
}

static inline float lsm6dsox_get_gyro_scale(lsm6dsox_gyro_range_t range)
{
  return lsm6dsox_get_gyro_range(range) / LSM6DSOX_OUTPUT_RES; // dps per LSB
}

static inline float lsm6dsox_get_accel_scale(lsm6dsox_accel_range_t range)
{
  return lsm6dsox_get_accel_range(range) / LSM6DSOX_OUTPUT_RES; // g per LSB
}

static inline void lsm6dsox_convert_temp(int16_t raw_temp, float *temp)
{
  // From datasheet: 256 LSB/degC, 0 degC = 25 degC
  *temp = (raw_temp / 256.0) + 25.0;
}

static inline void lsm6dsox_convert_gyro(
    int16_t *raw_gyro, float *gyro, lsm6dsox_gyro_range_t range)
{
  float scale = lsm6dsox_get_gyro_scale(range); // dps per LSB
  for (int i = 0; i < 3; i++)
  {
    gyro[i] = raw_gyro[i] * scale; // dps
  }
}

static inline void lsm6dsox_convert_accel(
    int16_t *raw_accel, float *accel, lsm6dsox_accel_range_t range)
{
  float scale = lsm6dsox_get_accel_scale(range); // g per LSB
  for (int i = 0; i < 3; i++)
  {
    accel[i] = raw_accel[i] * scale; // g
  }
}

static inline int lsm6dsox_init(lsm6dsox_dev_t *dev)
{
  if (!dev)
    return -1;

  dev->initialized = false;

  // Check WHO_AM_I
  uint8_t whoami;
  if (dev->read(dev->handle, LSM6DSOX_WHOAMI, &whoami, 1) != 0)
  {
    return -1;
  }

  if (whoami != 0x6C)
  {
    return -2; // Wrong device
  }

  // Configure registers
  uint8_t ctrl1 = lsm6dsox_get_ctrl1(dev->accel_odr, dev->accel_range, false); // No LPF2
  dev->write(dev->handle, LSM6DSOX_CTRL1_XL, &ctrl1, 1);
  uint8_t ctrl2 = lsm6dsox_get_ctrl2(dev->gyro_odr, dev->gyro_range);
  dev->write(dev->handle, LSM6DSOX_CTRL2_G, &ctrl2, 1);
  uint8_t ctrl8 = lsm6dsox_get_ctrl8(false, LSM6DSOX_HPF_ODR_DIV_4); // No HPF
  dev->write(dev->handle, LSM6DSOX_CTRL8_XL, &ctrl8, 1);
  dev->initialized = true;
  return 0;
}

static inline int lsm6dsox_read(
    const lsm6dsox_dev_t *dev, lsm6dsox_data_t *data)
{
  if (!dev || !dev->initialized)
    return -1;

  int16_t raw[7];
  uint8_t *buffer = data->buffer;

  int ret = dev->read(dev->handle, LSM6DSOX_OUT_TEMP_L, data->buffer, 14);
  if (ret != 0)
    return ret;

  // Combine low and high bytes (little-endian)
  for (int i = 0; i < 7; i++)
  {
    raw[i] = (int16_t)(buffer[2 * i + 1] << 8 | buffer[2 * i]);
  }

  lsm6dsox_convert_temp(raw[0], &data->temp);
  lsm6dsox_convert_gyro(&raw[1], data->gyro, dev->gyro_range);
  lsm6dsox_convert_accel(&raw[4], data->accel, dev->accel_range);

  // Flip axes to forward-right-down convention
  // See Figure 4 in datasheet
  data->accel[1] = -data->accel[1];
  data->gyro[1] = -data->gyro[1];
  data->accel[2] = -data->accel[2];
  data->gyro[2] = -data->gyro[2];

  // Convert gyro dps → rad/s
  for (int i = 0; i < 3; i++) {
      data->gyro[i] *= 0.0174533f;
  }

  // Apply calibration biases
  for (int i = 0; i < 3; i++)
  {
    data->accel[i] -= dev->accel_bias[i];
    data->gyro[i] -= dev->gyro_bias[i];
  }

  return 0;
}

#endif // LSM6DSOX_H