
#include <string.h>
#include "combine_func.h"

int combine_func_init(combine_func_arg_t* arg, combine_func_res_t* res, combine_func_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    /* Initialize inputs */
    memset(arg, 0, sizeof(combine_func_arg_t));

    /* Initialize outputs */
    memset(res, 0, sizeof(combine_func_res_t));

    /* Nonzero assignments */
    arg->c1.data[0] = 1.000000f;
    arg->c1.data[1] = 2.000000f;
    arg->c1.data[2] = 3.000000f;
    arg->c2.data[0] = 4.000000f;
    arg->c2.data[1] = 5.000000f;

    _Static_assert(sizeof(combine_func_arg_t) == 5 * sizeof(float),
        "Non-contiguous arg struct; please enable -fpack-struct or equivalent.");

    _Static_assert(sizeof(combine_func_res_t) == 17 * sizeof(float),
        "Non-contiguous res struct; please enable -fpack-struct or equivalent.");

    return 0;
}

int combine_func_step(combine_func_arg_t* arg, combine_func_res_t* res, combine_func_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    // Marshal inputs to CasADi format
    const float* kernel_arg[combine_func_SZ_ARG];
    kernel_arg[0] = (float*)&arg->c1;
    kernel_arg[1] = (float*)&arg->c2;

    // Marshal outputs to CasADi format
    float* kernel_res[combine_func_SZ_RES];
    kernel_res[0] = (float*)&res->combined;
    kernel_res[1] = (float*)&res->c2_out;
    kernel_res[2] = (float*)&res->extra1;
    kernel_res[3] = (float*)&res->extra2;

    // Call kernel function
    return combine_func(kernel_arg, kernel_res, work->iw, work->w, 0);
}