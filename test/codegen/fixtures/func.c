
#include <string.h>
#include "func.h"

int func_init(func_arg_t* arg, func_res_t* res, func_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    /* Initialize inputs */
    memset(arg, 0, sizeof(func_arg_t));

    /* Initialize outputs */
    memset(res, 0, sizeof(func_res_t));

    /* Nonzero assignments */
    arg->x[0] = 1.000000f;
    arg->x[1] = 2.000000f;
    arg->y = 3.000000f;

    _Static_assert(sizeof(func_arg_t) == 3 * sizeof(float),
        "Non-contiguous arg struct; please enable -fpack-struct or equivalent.");

    _Static_assert(sizeof(func_res_t) == 4 * sizeof(float),
        "Non-contiguous res struct; please enable -fpack-struct or equivalent.");

    return 0;
}

int func_step(func_arg_t* arg, func_res_t* res, func_work_t* work) {
    if (!arg || !res || !work) {
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
    return func(kernel_arg, kernel_res, work->iw, work->w, 0);
}