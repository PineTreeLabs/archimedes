// gcc main.c pid.c

#include "pid.h"

// PROTECTED-REGION-START: imports
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

    const int n_iter = 100;
    const int n_w = 100000;
    double times[n_iter];
    clock_t start, end;
    double cpu_time_used;

    printf("Running %d iterations and measuring execution time...\n", n_iter);
    for (int i = 0; i < n_iter; i++) {
        start = clock();

        // Do work  --->
        for (int j = 0; j < n_w; j++) {
            pid(arg, res, iw, w, 0);
    
            for (int i = 0; i < sizeof(x) / sizeof(x[0]); i++) {
                x[i] = x_new[i];
            }
        }
        // <--- Do work

        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        times[i] = cpu_time_used / n_w;
    }
    
    // Calculate mean
    double sum = 0.0;
    for (int i = 0; i < n_iter; i++) {
        sum += times[i];
    }
    double mean = sum / n_iter;
    
    // Calculate standard deviation
    double sum_squared_diff = 0.0;
    for (int i = 0; i < n_iter; i++) {
        double diff = times[i] - mean;
        sum_squared_diff += diff * diff;
    }
    double std_dev = sqrt(sum_squared_diff / n_iter);
    
    // Report results
    printf("\nExecution Time Statistics:\n");
    printf("Mean: %.6e μs\n", mean * 1e6);
    printf("Standard Deviation: %.6e μs\n", std_dev * 1e6 );

    // PROTECTED-REGION-END

    return 0;
}