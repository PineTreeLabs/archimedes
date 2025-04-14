#include <Arduino.h>
#include <TimerOne.h>
#include "pid.h"

// Control frequency: 100 Hz
const unsigned long CONTROL_PERIOD_US = 10000; // 10,000 microseconds = 10ms = 100Hz

// Allocate memory for the PID
float x[3] = {0.0, 0.0, 0.0};  // State vector (integral, previous error, etc.)
float e = 0.0;                  // Error signal
float x_new[3] = {0.0, 0.0, 0.0}; // Updated state
float u = 0.0;                  // Control signal

// Prepare pointers to inputs and outputs
const float* arg[pid_SZ_ARG] = {0};
float* res[pid_SZ_RES] = {0};

// Allocate work vectors
int iw[pid_SZ_IW];
float w[pid_SZ_W];

// Variables for setpoint and measurement
float setpoint = 0.0;
float measurement = 0.0;

// For timing verification (optional)
volatile unsigned long last_execution_time = 0;
volatile unsigned long execution_duration = 0;

// Flag to indicate when control loop has run
volatile bool control_loop_complete = false;


// Timer interrupt handler - runs at precisely 100Hz
void controlLoop() {
  unsigned long start_time = micros();

  // Calculate error
  e = setpoint - measurement;

  // Run the PID controller
  pid(arg, res, iw, w, 0);

  // ...do something with the control signal

  // Update state for next iteration
  for (int i = 0; i < 3; ++i) {
    x[i] = x_new[i];
  }
  
  // Timing diagnostics
  execution_duration = micros() - start_time;
  last_execution_time = millis();
  control_loop_complete = true;
}

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Set pointers once
  arg[0] = x;
  arg[1] = &e;
  res[0] = x_new;
  res[1] = &u;

  // Initialize Timer1 for precise 100Hz (10ms) timing
  Timer1.initialize(CONTROL_PERIOD_US);
  Timer1.attachInterrupt(controlLoop);

  Serial.println(F("PID controller initialized at 100Hz"));
  Serial.println(F("Format: time,setpoint,measurement,control_signal,execution_time_us"));
}

void loop() {
  // ... read sensors for setpoint and measurement

  // Output data for monitoring if control loop has completed
  if (control_loop_complete) {
    control_loop_complete = false;
    
    // Print data in CSV format for easy plotting
    Serial.print(last_execution_time);
    Serial.print(F(","));
    Serial.print(setpoint, 3);
    Serial.print(F(","));
    Serial.print(measurement, 3);
    Serial.print(F(","));
    Serial.print(u, 3);
    Serial.print(F(","));
    Serial.println(execution_duration);
  }
  
  // Small delay to avoid flooding the serial port
  delay(50);

}
