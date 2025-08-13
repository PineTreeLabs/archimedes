#include <Arduino.h>
#include <TimerOne.h>
#include "iir_filter.h"

// Sampling rate: 100 Hz
const unsigned long SAMPLE_RATE_US = 10000;

// Declare the input, output, and workspace structures
iir_filter_arg_t arg;
iir_filter_res_t res;
iir_filter_work_t work;

int n = sizeof(arg.b) / sizeof(arg.b[0]) - 1;  // Filter order
float y;  // Output, to be fed to actuator, control algorithm, etc.

// Timer interrupt handler
void timerInterrupt() {
    arg.u = 1.0;  // Or read from sensor, other algorithm output, etc.
    iir_filter_step(&arg, &res, &work);  // Call IIR filter
    y = res.y_hist[0];  // Output, to be fed to actuator, control algorithm, etc.

    // Copy output arrays back to inputs
    for (int j = 0; j < n; j++) {
        arg.u_prev[j] = res.u_hist[j];
        arg.y_prev[j] = res.y_hist[j];
    }
    arg.u_prev[n] = res.u_hist[n];
}

void setup(){
    // Initialize the structs
    iir_filter_init(&arg, &res, &work);

    Serial.begin(9600);

    // Initialize Timer1 for interrupts at 100 Hz
    Timer1.initialize(SAMPLE_RATE_US);
    Timer1.attachInterrupt(timerInterrupt);
}

void loop() {
    // ...non-time-critical tasks
    delay(10);
}