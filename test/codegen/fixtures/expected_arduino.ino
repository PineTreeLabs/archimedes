#include <Arduino.h>
#include <TimerOne.h>
#include "gen.h"

// PROTECTED-REGION-START: imports
// ... User-defined imports and includes
// PROTECTED-REGION-END

// Sampling rate:  Hz
const unsigned long SAMPLE_RATE_US = ;

// Allocate memory for inputs and outputs
float x[2] = {1.0, 2.0};
float y = 3.0;

float x_new[2] = {0};
float z[2] = {0};

// Prepare pointers to inputs, outputs, and work arrays
const float* arg[test_func_SZ_ARG] = {0};
float* res[test_func_SZ_RES] = {0};
int iw[test_func_SZ_IW];
float w[test_func_SZ_W];

// Flag for interrupt timer
volatile bool control_loop_flag = false;

// PROTECTED-REGION-START: allocation
// ... User-defined memory allocation and function declaration
// PROTECTED-REGION-END

// Timer interrupt handler
void timerInterrupt() {
    // PROTECTED-REGION-START: interrupt
    // Set flag for main loop to run control function
    control_loop_flag = true;
    // PROTECTED-REGION-END
}

void setup(){
    // Set up input and output pointers
    arg[0] = x;
    arg[1] = &y;

    res[0] = x_new;
    res[1] = z;

    // PROTECTED-REGION-START: setup
    // ... User-defined setup code
    Serial.begin(9600);
    // PROTECTED-REGION-END

    // Initialize Timer1 for interrupts at  Hz
    Timer1.initialize(SAMPLE_RATE_US);
    Timer1.attachInterrupt(timerInterrupt);
}

void loop() {
    // Check if control loop should run (set by timer interrupt)
    if (control_loop_flag) {
        control_loop_flag = false;
        
        // PROTECTED-REGION-START: control_loop
        // ... User-defined timed code
        test_func(arg, res, iw, w, 0);
        // PROTECTED-REGION-END
    }
    
    // PROTECTED-REGION-START: loop
    // ... User-defined non-time-critical tasks
    delay(10);
    // PROTECTED-REGION-END
}