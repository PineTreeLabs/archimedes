
#include "func.h"

int func_init(func_arg_t* arg, func_res_t* res, func_workspace_t* workspace) {
    if (!arg || !res || !workspace) {
        return -1; // Invalid pointers
    }

    // Initialize inputs
    arg->x[0] = 1.0;
    arg->x[1] = 2.0;

    arg->y = 3.0;

    
    // Initialize outputs
    memset(res, 0, sizeof(*res));

    return 0;
}

int func_step(func_arg_t* arg, func_res_t* res, func_workspace_t* workspace) {
    if (!arg || !res || !workspace) {
        return -1; // Invalid pointers
    }
    
    // Marshal inputs to CasADi format
    const float* kernel_arg[func_SZ_ARG];
    kernel_arg[0] = arg->x;
    kernel_arg[1] = &arg->y;
    
    // Marshal outputs to CasADi format
    float* kernel_res[func_SZ_RES];
    kernel_res[0] = res->x_new;
    kernel_res[1] = res->z;
    
    // Call kernel function
    return func(kernel_arg, kernel_res, workspace->iw, workspace->w, 0);
}