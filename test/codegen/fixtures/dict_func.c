
#include <string.h>
#include "dict_func.h"

int dict_func_init(dict_func_arg_t* arg, dict_func_res_t* res, dict_func_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    /* Initialize inputs */
    memset(arg, 0, sizeof(dict_func_arg_t));

    /* Initialize outputs */
    memset(res, 0, sizeof(dict_func_res_t));

    /* Nonzero assignments */
    arg->config.lr = 0.010000f;
    arg->config.momentum = 0.900000f;
    arg->bounds[1] = 1.000000f;
    arg->single_tuple[0] = 42.000000f;

    return 0;
}

int dict_func_step(dict_func_arg_t* arg, dict_func_res_t* res, dict_func_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    // Marshal inputs to CasADi format
    const float* kernel_arg[dict_func_SZ_ARG];
    kernel_arg[0] = &arg->config.lr;
    kernel_arg[1] = &arg->config.momentum;
    kernel_arg[2] = &arg->bounds[0];
    kernel_arg[3] = &arg->bounds[1];
    kernel_arg[4] = &arg->single_tuple[0];
    
    // Marshal outputs to CasADi format
    float* kernel_res[dict_func_SZ_RES];
    kernel_res[0] = &res->output.result;
    
    // Call kernel function
    return dict_func(kernel_arg, kernel_res, work->iw, work->w, 0);
}