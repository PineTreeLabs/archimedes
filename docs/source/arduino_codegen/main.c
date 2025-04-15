// gcc main.c pid.c -o main
#include <stdio.h>
#include "pid.h"

// Allocate memory for inputs and outputs
float x[3] = {1.0, 2.0, 3.0};  // State vector
float e = 0.5;  // Error signal (scalar)

float x_new[3] = {0};  // Updated state
float u = {0};  // Output vector (scalar)

// Prepare pointers to inputs, outputs, and work arrays
const float* arg[pid_SZ_ARG] = {0};
float* res[pid_SZ_RES] = {0};
int iw[pid_SZ_IW];
float w[pid_SZ_W];

int main(int argc, char *argv[]) {
    // Set up input arguments
    arg[0] = x;
    arg[1] = &e;
    
    // Set up output arguments
    res[0] = x_new;
    res[1] = &u;

    // Call the function
    pid(arg, res, iw, w, 0);

    // Print results
    printf("x_new[0] = %f\n", x_new[0]);
    printf("x_new[1] = %f\n", x_new[1]);
    printf("x_new[2] = %f\n", x_new[2]);
    printf("u = %f\n", u);
    
    // Copy output to input (for next iteration)
    for (int i = 0; i < 3; ++i) {
        x[i] = x_new[i];
    }
    
    return 0;
}