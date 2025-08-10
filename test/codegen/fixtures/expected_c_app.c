// gcc test_app.c gen.c

#include "gen.h"

// PROTECTED-REGION-START: imports
// ... User-defined imports and includes
// PROTECTED-REGION-END

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

// PROTECTED-REGION-START: allocation
// ... User-defined memory allocation and function declaration
// PROTECTED-REGION-END

int main(int argc, char *argv[]) {
    // Set up input and output pointers
    arg[0] = x;
    arg[1] = &y;

    res[0] = x_new;
    res[1] = z;

    // PROTECTED-REGION-START: main
    // ... User-defined program body
    test_func(arg, res, iw, w, 0);
    // PROTECTED-REGION-END

    return 0;
}