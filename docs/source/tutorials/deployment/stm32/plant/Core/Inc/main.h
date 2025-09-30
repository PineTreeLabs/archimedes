/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */
void sample_callback(void);
/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define ENC_OUTB_Pin GPIO_PIN_7
#define ENC_OUTB_GPIO_Port GPIOA
#define Driver_ENB_Pin GPIO_PIN_15
#define Driver_ENB_GPIO_Port GPIOF
#define Driver_ENA_Pin GPIO_PIN_13
#define Driver_ENA_GPIO_Port GPIOE
#define ENC_OUTA_Pin GPIO_PIN_14
#define ENC_OUTA_GPIO_Port GPIOD
#define Driver_INA_Pin GPIO_PIN_9
#define Driver_INA_GPIO_Port GPIOG
#define Driver_INB_Pin GPIO_PIN_14
#define Driver_INB_GPIO_Port GPIOG
#define Onboard_LED_Pin GPIO_PIN_7
#define Onboard_LED_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
