// gcc main.c pid.c -o main.c

#include "pid.h"

// PROTECTED-REGION-START: imports
// ... User-defined imports and includes
// PROTECTED-REGION-END

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

// PROTECTED-REGION-START: allocation
// ... User-defined memory allocation and function declaration
// PROTECTED-REGION-END

int main(int argc, char *argv[]) {
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

    // PROTECTED-REGION-START: main
// ... User-defined program body
    pid(arg, res, iw, w, 0);
    // PROTECTED-REGION-END

    return 0;
}