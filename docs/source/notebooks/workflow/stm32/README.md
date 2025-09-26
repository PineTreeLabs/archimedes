# HIL configuration

## Plant board

### Pin configuration

| STM32 pin | Nucleo pin | Function    | Notes                               |
|-----------|------------|-------------|-------------------------------------|
| PA3       | A0         | ADC1_IN3    | ADC1 rank 1 (PWM input)             |
| PA4       | D24        | DAC_OUT1    | V_CS output                         |
| PA5       | D13        | DAC_OUT2    | VOUTA output                        |
| PD14      | D10        | GPIO_Output | Encoder output A                    |
| PA7       | D11        | GPIO_Output | Encoder output B                    |
| PB7       | N/A        | GPIO_Output | Onboard blue LED                    |
| PG9       | D0         | GPIO_Input  | Driver INA                          |
| PG14      | D1         | GPIO_Input  | Driver INB                          |
| PF15      | D2         | GPIO_Input  | Driver ENA                          |
| PE13      | D3         | GPIO_Input  | Driver ENB                          |

### Peripheral configuration

- ADC1 on TIM3 at 10 kHz
- DAC_OUT1, DAC_OUT2, and control loop on TIM6 at 10 kHz

**CLOCK**

Set High Speed Clock (HSE) to Crystal/Ceramic Resonator

- PLL Source Mux: HSE
- M = 8, N = 360, P = 2
- System Clock Mux: PLLCLK
- AHB Prescaler: 1
- APB1 Prescaler: 4
- APB2 Prescaler: 2

Provides APB1 timers (TIM1) 90 MHz, APB2 timers (TIM2, TIM3, TIM6) 180 MHz.
See Figure 4 in STM32F439xx datasheet for block diagram for the clocks.

**TIM6**

- 10 kHz: PSC = 179, ARR = 49
- Enable global interrupt
- Trigger Event Selection: Update Event

**DAC**

- Circular mode
- Half-word data width
- Increment peripheral address: NO
- Increment memory address: YES
- Trigger: TIM6 Trigger Out Event

**ADC1**

- Clock prescaler: PCLK2 divided by 4
- Resolution: 12 bits
- Scan Conversion: Enabled
- Continuous Conversion: Enabled
- DMA Continuous Requests: Enabled
- External Trigger Conversion Source: TIM3 Trigger Out Event
- External Trigger Conversion Edge: Rising Edge
- 84 cycle sample time (all channels)

## Controller board


### Pin configuration

| STM32 pin | Nucleo pin | Function    | Notes                               |
|-----------|------------|-------------|-------------------------------------|
| PF3       | A3         | ADC3_IN9    | ADC3 rank 2 (VOUTB)                 |
| PC3       | A2         | ADC3_IN13   | ADC3 rank 1 (VOUTA)                 |
| PC0       | A1         | ADC1_IN10   | ADC1 rank 2 (V_CS)                  |
| PA0       | D32        | TIM2_CH1    | Encoder input A                     |
| PB3       | D23        | TIM2_CH2    | Encoder input B                     |
| PE11      | N/A        | GPIO_Input  | Onboard button                      |
| PB7       | N/A        | GPIO_Output | Onboard blue LED                    |
| PG9       | D0         | GPIO_Output | Driver INA                          |
| PG14      | D1         | GPIO_Output | Driver INB                          |
| PF15      | D2         | GPIO_Output | Driver ENA (output open drain)      |
| PE13      | D3         | GPIO_Output | Driver ENB (output open drain)      |
| PE9       | D6         | TIM1_CH1    | PWM output                          |


### Peripheral configuration

- PWM on TIM1 at 20 kHz
- Quadrature encoder reading on TIM2
- ADC and control loop on TIM3 at 10 kHz

**CLOCK**

Set High Speed Clock (HSE) to Crystal/Ceramic Resonator

- PLL Source Mux: HSE
- M = 8, N = 360, P = 2
- System Clock Mux: PLLCLK
- AHB Prescaler: 1
- APB1 Prescaler: 4
- APB2 Prescaler: 2

Provides APB1 timers (TIM1) 90 MHz, APB2 timers (TIM2, TIM3, TIM6) 180 MHz.
See Figure 4 in STM32F439xx datasheet for block diagram for the clocks.

**TIM1**

- Clock Source: Disable
- Channel 1: PWM Generation CH1
- 20 kHz: PSC = 8, ARR = 999
- Enable global interrupt
- Trigger Event Selection: Update Event

**TIM2**

- Combined Channels: Encoder Mode
- Encoder Mode: TI1 and TI2 (rising edge)

**TIM3**

- Clock Source: Internal Clock
- 10 kHz: PSC = 179, ARR = 49
- Enable global interrupt
- Trigger Event Selection: Update Event

**TIM6**

- 10 kHz: PSC = 179, ARR = 49
- Enable global interrupt
- Trigger Event Selection: Update Event

**DAC**

- Circular mode
- Half-word data width
- Increment peripheral address: NO
- Increment memory address: YES
- Trigger: TIM6 Trigger Out Event

**ADC1/3**

- Clock prescaler: PCLK2 divided by 8
- Resolution: 12 bits
- Scan Conversion: Enabled
- Continuous Conversion: Enabled
- DMA Continuous Requests: Enabled
- External Trigger Conversion Source: TIM3 Trigger Out Event
- External Trigger Conversion Edge: Rising Edge
- 84 cycle sample time (all channels)

**USART3**

- PD8 for TX, PD9 for RX (uses CN1 USB)
- De-solder SB4 and SB7 to enable CN1 USB comm
- Asynchronous mode
- 115200 bit/s baud rate
- 8 bit word length


## Interface


### RC filters

Used to convert PWM signal to analog, and for low-pass filtering.

1k resistor (R1) + 100 nF capacitor (C1) = 1 ms time constant (~1.5 kHz cutoff)


### Wiring

| Controller |  Plant  | Notes                               |
|------------|---------|-------------------------------------|
| D6         | R1_HIGH | RC filter 1 (PWM from controller)   |
| C1_HIGH    | A0      | RC filter 1 (PWM to plant)          |
| C1_LOW     | GND     | RC filter 1                         |
| R1_LOW     | C1_HIGH | RC filter 1                         |
| 5V         | E5V     | Power to plant board                |
| GND        | GND     | Connect grounds between boards      |
| A1         | D24     | A/D for V_CS                        |
| R2_HIGH    | D13     | RC filter 2 (VOUTA from plant)      |
| A2         | C2_HIGH | RC filter 2 (VOUTA to controller)   |
| C2_LOW     | GND     | RC filter 2                         |
| R2_LOW     | C2_HIGH | RC filter 2                         |
| A3         | GND     | A/D for VOUTB (0V)                  |
| D32        | D10     | ENCA signal                         |
| D23        | D11     | ENCB signal                         |
| D0         | D0      | INA signal                          |
| D1         | D1      | INB signal                          |
| D2         | D2      | ENA signal                          |
| D3         | D3      | ENB signal                          |


## Deployment

Assuming CubeMX has already been used to generate the project code, and that the C applications have not been changed, the code can be deployed as follows:

1. Run the Jupyter notebooks to generate code for the plant model and controller
2. Deploy the plant model:

	```bash
	cd plant
	cmake --preset=Release
	cmake --build --preset=Release
	openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
	        -c "program build/Release/motor_plant.elf verify reset exit"
	cd ..
	```

3. Deploy the controller:

	```bash
	cd plant
	cmake --preset=Release
	cmake --build --preset=Release
	openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
	        -c "program build/Release/motor_controller.elf verify reset exit"
	```

4. Collect data with the Python serial script:

	```bash
	cd ..
	python serial_receive.py --save hil_data.csv
	```

