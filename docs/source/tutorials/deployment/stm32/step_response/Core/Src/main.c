/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
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
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

#define ADC1_CHANNELS 1
#define ADC3_CHANNELS 2
#define SAMPLE_COUNT 2000 // How many samples to collect

enum ExptType
{
    STEP_RESPONSE,
    RAMP_RESPONSE
};

typedef struct
{
    uint32_t sample_idx;  // Sample index into logger arrays
    uint32_t loop_count;  // How many times we've actually sampled at 10 kHz
    uint16_t sample_rate; // loop_counts per sample_idx
    uint16_t pwm_duty[SAMPLE_COUNT];
    int32_t v_motor_mv[SAMPLE_COUNT];
    uint16_t i_motor_ma[SAMPLE_COUNT];
    int32_t pos_mdeg[SAMPLE_COUNT];
} log_data_t;

typedef union
{
    uint32_t buffer[ADC1_CHANNELS];
    struct
    {
        uint32_t cs_raw;
    } channels;
} adc1_data_t;

typedef union
{
    uint32_t buffer[ADC3_CHANNELS];
    struct
    {
        uint32_t vout_a_raw;
        uint32_t vout_b_raw;
    } channels;
} adc3_data_t;

typedef struct
{
    int32_t cur_count;  // Current encoder position count
    int32_t prev_count; // Temporary value used to calculate updates
    float pos_deg;      // Integrated position [deg]
} enc_data_t;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#define PWM_COUNT 1000
#define ENC_COUNT 65536

#define ADC_TO_VOLTS (3.3f / 4095.0f)          // Convert ADC counts to voltage
#define CS_V_PER_AMP (0.14f)                   // VNH5019 spec: 0.14 V/A
#define CS_SCALE (ADC_TO_VOLTS / CS_V_PER_AMP) // ADC counts to amperage

#define ENC_CPR        48      // Counts per motor revolution (datasheet)
#define GEAR_RATIO     46.8512f      // 47:1 gearbox
#define RAD_PER_COUNT  (2.0f * M_PI / (ENC_CPR * GEAR_RATIO))
#define RAD_TO_DEG     (180.0f / M_PI)
#define DEG_PER_COUNT  (360.0f / (ENC_CPR * GEAR_RATIO))  // DEBUG: Shouldn't need this

#define ENC_VOUT_R1 (47.0f) // First leg of voltage divider
#define ENC_VOUT_R2 (15.0f) // Second leg of voltage divider
#define ENC_VOUT_SCALE ((3.3f / 4095.0f) * (ENC_VOUT_R1 + ENC_VOUT_R2) / ENC_VOUT_R2)

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
ADC_HandleTypeDef hadc3;
DMA_HandleTypeDef hdma_adc1;
DMA_HandleTypeDef hdma_adc3;

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;

UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */
enum ExptType expt_type = STEP_RESPONSE;
// enum ExptType expt_type = RAMP_RESPONSE;

volatile bool sample_flag = false;
volatile adc1_data_t adc1_data;
volatile adc3_data_t adc3_data;

uint16_t pwm_duty = 0;
enc_data_t enc_data = {ENC_COUNT / 2, ENC_COUNT / 2, 0.0f};
log_data_t log_data = {0};
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM3_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM2_Init(void);
static void MX_ADC3_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void)
{

    /* USER CODE BEGIN 1 */

    /* USER CODE END 1 */

    /* MCU Configuration--------------------------------------------------------*/

    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();

    /* USER CODE BEGIN Init */

    /* USER CODE END Init */

    /* Configure the system clock */
    SystemClock_Config();

    /* USER CODE BEGIN SysInit */

    /* USER CODE END SysInit */

    /* Initialize all configured peripherals */
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_ADC1_Init();
    MX_TIM3_Init();
    MX_USART3_UART_Init();
    MX_TIM1_Init();
    MX_TIM2_Init();
    MX_ADC3_Init();
    /* USER CODE BEGIN 2 */

    // Start PWM output
    if (HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1) != HAL_OK)
        Error_Handler();

    // Start ADC with DMA in circular mode
    if (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)adc1_data.buffer, ADC1_CHANNELS) != HAL_OK)
        Error_Handler();
    if (HAL_ADC_Start_DMA(&hadc3, (uint32_t *)adc3_data.buffer, ADC3_CHANNELS) != HAL_OK)
        Error_Handler();

    // Configure motor direction forward (ccw)
    motor_fwd();

    // Enable motor driver
    HAL_GPIO_WritePin(Motor_ENA_GPIO_Port, Motor_ENA_Pin, GPIO_PIN_SET); // ENA/DIAGA = HIGH (enable)
    HAL_GPIO_WritePin(Motor_ENB_GPIO_Port, Motor_ENB_Pin, GPIO_PIN_SET); // ENB/DIAGB = HIGH (enable)

    // Start encoder timer
    if (HAL_TIM_Encoder_Start(&htim2, TIM_CHANNEL_ALL) != HAL_OK)
        Error_Handler();

    // Reset encoder counter to middle value
    __HAL_TIM_SET_COUNTER(&htim2, enc_data.cur_count);

    // Start control timer
    if (HAL_TIM_Base_Start_IT(&htim3) != HAL_OK)
        Error_Handler();

    /* USER CODE END 2 */

    /* Infinite loop */
    /* USER CODE BEGIN WHILE */

    HAL_GPIO_WritePin(Onboard_LED_GPIO_Port, Onboard_LED_Pin, GPIO_PIN_SET);
    HAL_Delay(1000); // Wait 1 sec for everything to come online
    HAL_GPIO_WritePin(Onboard_LED_GPIO_Port, Onboard_LED_Pin, GPIO_PIN_RESET);

    // Wait for the USER button to be pressed
    while (1)
    {
        if (HAL_GPIO_ReadPin(Onboard_Button_GPIO_Port, Onboard_Button_Pin))
            break; // Continue on to main control loop
        HAL_Delay(1);
    }
    // Turn on blue LED to indicate start of test
    HAL_GPIO_WritePin(Onboard_LED_GPIO_Port, Onboard_LED_Pin, GPIO_PIN_SET);

    /* STEP RESPONSE */
    uint32_t next_incr = 0; // Loop count to increment duty cycle

    switch (expt_type)
    {
    case STEP_RESPONSE:
        log_data.sample_rate = 1; // Sample at 10 kHz (200 ms total)
        pwm_duty = PWM_COUNT / 2; // 50% duty cycle
        break;
    case RAMP_RESPONSE:
        log_data.sample_rate = 100; // Sample at 100 Hz (20s total)
        pwm_duty = 0;               // Start at 0% duty cycle
        break;
    }

    // Reset indices
    log_data.sample_idx = 0;
    log_data.loop_count = 0; // Number of 10 kHz updates per sample
    sample_flag = false;
    motor_set();

    while (1)
    {
        if (log_data.sample_idx >= SAMPLE_COUNT)
            break;
        /* USER CODE END WHILE */

        /* USER CODE BEGIN 3 */

        switch (expt_type)
        {
        case RAMP_RESPONSE:
            // Increase PWM duty cycle by 5% every 500 ms up to 100%, then
            // ramp back down
            if (log_data.loop_count >= next_incr)
            {
                if (log_data.sample_idx < SAMPLE_COUNT / 2)
                    pwm_duty += PWM_COUNT / 20;
                else
                    pwm_duty -= PWM_COUNT / 20;

                next_incr += log_data.sample_rate * (SAMPLE_COUNT / 40);
                motor_set();
            }
            break;
        default:
            break;
        }

        if (sample_flag)
            sample_callback();
    }

    pwm_duty = 0; // Turn OFF motor
    motor_set();
    send_data();
    HAL_GPIO_WritePin(Onboard_LED_GPIO_Port, Onboard_LED_Pin, GPIO_PIN_RESET);

    return 0;

    /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    /** Configure the main internal regulator output voltage
     */
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    /** Initializes the RCC Oscillators according to the specified parameters
     * in the RCC_OscInitTypeDef structure.
     */
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM = 8;
    RCC_OscInitStruct.PLL.PLLN = 360;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
    RCC_OscInitStruct.PLL.PLLQ = 4;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
        Error_Handler();
    }

    /** Activate the Over-Drive mode
     */
    if (HAL_PWREx_EnableOverDrive() != HAL_OK)
    {
        Error_Handler();
    }

    /** Initializes the CPU, AHB and APB buses clocks
     */
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
 * @brief ADC1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_ADC1_Init(void)
{

    /* USER CODE BEGIN ADC1_Init 0 */

    /* USER CODE END ADC1_Init 0 */

    ADC_ChannelConfTypeDef sConfig = {0};

    /* USER CODE BEGIN ADC1_Init 1 */

    /* USER CODE END ADC1_Init 1 */

    /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
     */
    hadc1.Instance = ADC1;
    hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
    hadc1.Init.Resolution = ADC_RESOLUTION_12B;
    hadc1.Init.ScanConvMode = ENABLE;
    hadc1.Init.ContinuousConvMode = ENABLE;
    hadc1.Init.DiscontinuousConvMode = DISABLE;
    hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_RISING;
    hadc1.Init.ExternalTrigConv = ADC_EXTERNALTRIGCONV_T3_TRGO;
    hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
    hadc1.Init.NbrOfConversion = 1;
    hadc1.Init.DMAContinuousRequests = ENABLE;
    hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
    if (HAL_ADC_Init(&hadc1) != HAL_OK)
    {
        Error_Handler();
    }

    /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
     */
    sConfig.Channel = ADC_CHANNEL_10;
    sConfig.Rank = 1;
    sConfig.SamplingTime = ADC_SAMPLETIME_84CYCLES;
    if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN ADC1_Init 2 */

    /* USER CODE END ADC1_Init 2 */
}

/**
 * @brief ADC3 Initialization Function
 * @param None
 * @retval None
 */
static void MX_ADC3_Init(void)
{

    /* USER CODE BEGIN ADC3_Init 0 */

    /* USER CODE END ADC3_Init 0 */

    ADC_ChannelConfTypeDef sConfig = {0};

    /* USER CODE BEGIN ADC3_Init 1 */

    /* USER CODE END ADC3_Init 1 */

    /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
     */
    hadc3.Instance = ADC3;
    hadc3.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
    hadc3.Init.Resolution = ADC_RESOLUTION_12B;
    hadc3.Init.ScanConvMode = ENABLE;
    hadc3.Init.ContinuousConvMode = ENABLE;
    hadc3.Init.DiscontinuousConvMode = DISABLE;
    hadc3.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_RISING;
    hadc3.Init.ExternalTrigConv = ADC_EXTERNALTRIGCONV_T3_TRGO;
    hadc3.Init.DataAlign = ADC_DATAALIGN_RIGHT;
    hadc3.Init.NbrOfConversion = 2;
    hadc3.Init.DMAContinuousRequests = ENABLE;
    hadc3.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
    if (HAL_ADC_Init(&hadc3) != HAL_OK)
    {
        Error_Handler();
    }

    /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
     */
    sConfig.Channel = ADC_CHANNEL_13;
    sConfig.Rank = 1;
    sConfig.SamplingTime = ADC_SAMPLETIME_84CYCLES;
    if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
    {
        Error_Handler();
    }

    /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
     */
    sConfig.Channel = ADC_CHANNEL_9;
    sConfig.Rank = 2;
    if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN ADC3_Init 2 */

    /* USER CODE END ADC3_Init 2 */
}

/**
 * @brief TIM1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_TIM1_Init(void)
{

    /* USER CODE BEGIN TIM1_Init 0 */

    /* USER CODE END TIM1_Init 0 */

    TIM_MasterConfigTypeDef sMasterConfig = {0};
    TIM_OC_InitTypeDef sConfigOC = {0};
    TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

    /* USER CODE BEGIN TIM1_Init 1 */

    /* USER CODE END TIM1_Init 1 */
    htim1.Instance = TIM1;
    htim1.Init.Prescaler = 8;
    htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim1.Init.Period = 999;
    htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim1.Init.RepetitionCounter = 0;
    htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    if (HAL_TIM_PWM_Init(&htim1) != HAL_OK)
    {
        Error_Handler();
    }
    sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
    sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
    if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
    {
        Error_Handler();
    }
    sConfigOC.OCMode = TIM_OCMODE_PWM1;
    sConfigOC.Pulse = 0;
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
    sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
    sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
    if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
    {
        Error_Handler();
    }
    sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
    sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
    sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
    sBreakDeadTimeConfig.DeadTime = 0;
    sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
    sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
    sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
    if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN TIM1_Init 2 */

    /* USER CODE END TIM1_Init 2 */
    HAL_TIM_MspPostInit(&htim1);
}

/**
 * @brief TIM2 Initialization Function
 * @param None
 * @retval None
 */
static void MX_TIM2_Init(void)
{

    /* USER CODE BEGIN TIM2_Init 0 */

    /* USER CODE END TIM2_Init 0 */

    TIM_Encoder_InitTypeDef sConfig = {0};
    TIM_MasterConfigTypeDef sMasterConfig = {0};

    /* USER CODE BEGIN TIM2_Init 1 */

    /* USER CODE END TIM2_Init 1 */
    htim2.Instance = TIM2;
    htim2.Init.Prescaler = 0;
    htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim2.Init.Period = 65535;
    htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
    sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
    sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
    sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
    sConfig.IC1Filter = 0;
    sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
    sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
    sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
    sConfig.IC2Filter = 0;
    if (HAL_TIM_Encoder_Init(&htim2, &sConfig) != HAL_OK)
    {
        Error_Handler();
    }
    sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
    sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
    if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN TIM2_Init 2 */

    /* USER CODE END TIM2_Init 2 */
}

/**
 * @brief TIM3 Initialization Function
 * @param None
 * @retval None
 */
static void MX_TIM3_Init(void)
{

    /* USER CODE BEGIN TIM3_Init 0 */

    /* USER CODE END TIM3_Init 0 */

    TIM_ClockConfigTypeDef sClockSourceConfig = {0};
    TIM_MasterConfigTypeDef sMasterConfig = {0};

    /* USER CODE BEGIN TIM3_Init 1 */

    /* USER CODE END TIM3_Init 1 */
    htim3.Instance = TIM3;
    htim3.Init.Prescaler = 179;
    htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim3.Init.Period = 49;
    htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
    {
        Error_Handler();
    }
    sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
    if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK)
    {
        Error_Handler();
    }
    sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
    sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
    if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN TIM3_Init 2 */

    /* USER CODE END TIM3_Init 2 */
}

/**
 * @brief USART3 Initialization Function
 * @param None
 * @retval None
 */
static void MX_USART3_UART_Init(void)
{

    /* USER CODE BEGIN USART3_Init 0 */

    /* USER CODE END USART3_Init 0 */

    /* USER CODE BEGIN USART3_Init 1 */

    /* USER CODE END USART3_Init 1 */
    huart3.Instance = USART3;
    huart3.Init.BaudRate = 115200;
    huart3.Init.WordLength = UART_WORDLENGTH_8B;
    huart3.Init.StopBits = UART_STOPBITS_1;
    huart3.Init.Parity = UART_PARITY_NONE;
    huart3.Init.Mode = UART_MODE_TX_RX;
    huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart3.Init.OverSampling = UART_OVERSAMPLING_16;
    if (HAL_UART_Init(&huart3) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN USART3_Init 2 */

    /* USER CODE END USART3_Init 2 */
}

/**
 * Enable DMA controller clock
 */
static void MX_DMA_Init(void)
{

    /* DMA controller clock enable */
    __HAL_RCC_DMA2_CLK_ENABLE();

    /* DMA interrupt init */
    /* DMA2_Stream0_IRQn interrupt configuration */
    HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);
    /* DMA2_Stream1_IRQn interrupt configuration */
    HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);
}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    /* USER CODE BEGIN MX_GPIO_Init_1 */

    /* USER CODE END MX_GPIO_Init_1 */

    /* GPIO Ports Clock Enable */
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOF_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(Motor_ENA_GPIO_Port, Motor_ENA_Pin, GPIO_PIN_RESET);

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(Motor_ENB_GPIO_Port, Motor_ENB_Pin, GPIO_PIN_RESET);

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(GPIOG, Motor_INA_Pin | Motor_INB_Pin, GPIO_PIN_RESET);

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(Onboard_LED_GPIO_Port, Onboard_LED_Pin, GPIO_PIN_RESET);

    /*Configure GPIO pin : Onboard_Button_Pin */
    GPIO_InitStruct.Pin = Onboard_Button_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(Onboard_Button_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pin : Motor_ENA_Pin */
    GPIO_InitStruct.Pin = Motor_ENA_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(Motor_ENA_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pin : Motor_ENB_Pin */
    GPIO_InitStruct.Pin = Motor_ENB_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(Motor_ENB_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pins : Motor_INA_Pin Motor_INB_Pin */
    GPIO_InitStruct.Pin = Motor_INA_Pin | Motor_INB_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

    /*Configure GPIO pin : Onboard_LED_Pin */
    GPIO_InitStruct.Pin = Onboard_LED_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(Onboard_LED_GPIO_Port, &GPIO_InitStruct);

    /* USER CODE BEGIN MX_GPIO_Init_2 */

    /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM3)
        sample_flag = true;
}

void motor_set(void)
{
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, pwm_duty);
}

/*
Reference:
|--INA--|--INB--|--STATE--|
|  LOW  | HIGH  | FORWARD | (CCW)
| HIGH  | HIGH  |  BRAKE  |
|  LOW  |  LOW  |  COAST  |
| HIGH  |  LOW  | REVERSE | (CW)
*/
void motor_rev(void)
{
    // Set motor direction forward (cw)
    HAL_GPIO_WritePin(Motor_INA_GPIO_Port, Motor_INA_Pin, GPIO_PIN_SET);   // INA = HIGH
    HAL_GPIO_WritePin(Motor_INB_GPIO_Port, Motor_INB_Pin, GPIO_PIN_RESET); // INB = LOW
}

void motor_fwd(void)
{
    // Set motor direction reverse (ccw)
    HAL_GPIO_WritePin(Motor_INA_GPIO_Port, Motor_INA_Pin, GPIO_PIN_RESET); // INA = LOW
    HAL_GPIO_WritePin(Motor_INB_GPIO_Port, Motor_INB_Pin, GPIO_PIN_SET);   // INB = HIGH
}

void sample_callback(void)
{
    // Read encoder and update position
    enc_data.cur_count = (int32_t)__HAL_TIM_GET_COUNTER(&htim2);
    
    // Handle encoder counter overflow/underflow by calculating the shortest path
    int32_t delta = enc_data.cur_count - enc_data.prev_count;
    if (delta > (ENC_COUNT / 2)) {
        delta -= ENC_COUNT;  // Wrapped backwards
    } else if (delta < -(ENC_COUNT / 2)) {
        delta += ENC_COUNT;  // Wrapped forwards
    }

    enc_data.pos_deg += (float)delta * DEG_PER_COUNT;
    enc_data.pos_deg = fmodf(enc_data.pos_deg + 360.0f, 360.0f);

    // Store for next iteration
    enc_data.prev_count = enc_data.cur_count;

    if ((log_data.loop_count % log_data.sample_rate == 0) && (log_data.sample_idx < SAMPLE_COUNT))
    {
        log_data.pwm_duty[log_data.sample_idx] = pwm_duty;
        log_data.v_motor_mv[log_data.sample_idx] = (int32_t)(((float)adc3_data.channels.vout_a_raw - (float)adc3_data.channels.vout_b_raw) * ENC_VOUT_SCALE * 1000); // millivolts
        log_data.i_motor_ma[log_data.sample_idx] = (uint16_t)(adc1_data.channels.cs_raw * CS_SCALE * 1000);                                                          // milliamps
        log_data.pos_mdeg[log_data.sample_idx] = (int32_t)(enc_data.pos_deg * 1000);                                                                                 // millidegrees
        log_data.sample_idx++;
    }

    log_data.loop_count++;
    sample_flag = false;
}

void send_data(void)
{
    char header[100];
    char buffer[200];

    snprintf(header, sizeof(header), "START,%lu,%u,%u,%lu\n",
             (uint32_t)SAMPLE_COUNT, PWM_COUNT, log_data.sample_rate, (uint32_t)0);
    HAL_UART_Transmit(&huart3, (uint8_t *)header, strlen(header), HAL_MAX_DELAY);

    // Send data in chunks to avoid USB buffer overflow
    for (uint32_t i = 0; i < SAMPLE_COUNT; i++)
    {
        snprintf(buffer, sizeof(buffer), "%lu,%u,%ld,%u,%ld\n",
                 i, log_data.pwm_duty[i], log_data.v_motor_mv[i],
                 log_data.i_motor_ma[i], log_data.pos_mdeg[i]);
        HAL_UART_Transmit(&huart3, (uint8_t *)buffer, strlen(buffer), HAL_MAX_DELAY);

        // Small delay to allow USB buffer to clear
        HAL_Delay(1);
    }

    HAL_UART_Transmit(&huart3, (uint8_t *)"END\n", 4, HAL_MAX_DELAY);
}

/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void)
{
    /* USER CODE BEGIN Error_Handler_Debug */
    /* User can add his own implementation to report the HAL error return state */
    __disable_irq();
    while (1)
    {
    }
    /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line)
{
    /* USER CODE BEGIN 6 */
    /* User can add his own implementation to report the file name and line number,
       ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
    /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
