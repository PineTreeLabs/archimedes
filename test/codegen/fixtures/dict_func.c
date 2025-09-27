
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

    _Static_assert(sizeof(dict_func_arg_t) == 5 * sizeof(float),
        "Non-contiguous arg struct; please enable -fpack-struct or equivalent.");

    _Static_assert(sizeof(dict_func_res_t) == 1 * sizeof(float),
        "Non-contiguous res struct; please enable -fpack-struct or equivalent.");

    return 0;
}

int dict_func_step(dict_func_arg_t* arg, dict_func_res_t* res, dict_func_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    // Marshal inputs to CasADi format
    const float* kernel_arg[dict_func_SZ_ARG];
    kernel_arg[0] = (float*)&arg->config;
    kernel_arg[1] = (float*)arg->bounds;
    kernel_arg[2] = (float*)arg->single_tuple;

    // Marshal outputs to CasADi format
    float* kernel_res[dict_func_SZ_RES];
    kernel_res[0] = (float*)&res->output;

    // Call kernel function
    return dict_func(kernel_arg, kernel_res, work->iw, work->w, 0);
}