#include <Arduino.h>
#include <TimerOne.h>
#include "pid.h"

// PROTECTED-REGION-START: imports
// ... User-defined imports and includes
// PROTECTED-REGION-END

// Sampling rate: 100 Hz
const unsigned long SAMPLE_RATE_US = 10000;

// Allocate memory for inputs and outputs
float x[3] = {1.0, 2.0, 3.0};  // State vector
float e = 0.5;  // Error signal (scalar)
float Kp = 1.0;  // Proportional gain
float Ki = 0.1;  // Integral gain
float Kd = 0.01;  // Derivative gain
float Ts = 0.01;  // Sampling time in seconds
float N = 100.0;  // Filter coefficient

float x_new[3] = {0};  // Updated state
float u = {0};  // Output vector (scalar)

// Prepare pointers to inputs, outputs, and work arrays
const float* arg[pid_SZ_ARG] = {0};
float* res[pid_SZ_RES] = {0};
int iw[pid_SZ_IW];
float w[pid_SZ_W];

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
    arg[1] = &e;
    arg[2] = &Kp;
    arg[3] = &Ki;
    arg[4] = &Kd;
    arg[5] = &Ts;
    arg[6] = &N;

    res[0] = x_new;
    res[1] = &u;

    // PROTECTED-REGION-START: setup
    // ... User-defined setup code
    Serial.begin(9600);
    // PROTECTED-REGION-END

    // Initialize Timer1 for interrupts at 100 Hz
    Timer1.initialize(SAMPLE_RATE_US);
    Timer1.attachInterrupt(timerInterrupt);
}

void loop() {
    // Check if control loop should run (set by timer interrupt)
    if (control_loop_flag) {
        control_loop_flag = false;
        
        // PROTECTED-REGION-START: control_loop
        // ... User-defined timed code
        pid(arg, res, iw, w, 0);

        for (int i = 0; i < 3; i++) {
            x[i] = x_new[i];
        }
        // PROTECTED-REGION-END
    }
    
    // PROTECTED-REGION-START: loop
    // ... User-defined non-time-critical tasks
    delay(100);
    Serial.print("u:");
    Serial.println(u);
    // PROTECTED-REGION-END
}