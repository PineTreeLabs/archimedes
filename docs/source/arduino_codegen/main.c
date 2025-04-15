// gcc pid.c pid.c -o pid
#include "pid.h"

// Allocate memory for inputs and outputs
float x[3] = {0.0, 0.0, 0.0};  // State vector
float e = 0.0;  // Error signal (scalar)

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
    
    return 0;
}