
#include <string.h>
#include "array_func.h"

int array_func_init(array_func_arg_t* arg, array_func_res_t* res, array_func_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    /* Initialize inputs */
    memset(arg, 0, sizeof(array_func_arg_t));

    /* Initialize outputs */
    memset(res, 0, sizeof(array_func_res_t));

    /* Nonzero assignments */
    arg->zero_d = 1.000000f;
    arg->one_d_single[0] = 1.000000f;
    arg->one_d_normal[1] = 1.000000f;
    arg->one_d_normal[2] = 2.000000f;
    arg->one_d_normal[3] = 3.000000f;
    arg->one_d_normal[4] = 4.000000f;
    arg->two_d_normal[0] = 1.000000f;
    arg->two_d_normal[1] = 4.000000f;
    arg->two_d_normal[2] = 2.000000f;
    arg->two_d_normal[3] = 5.000000f;
    arg->two_d_normal[4] = 3.000000f;
    arg->two_d_normal[5] = 6.000000f;
    arg->edge_case.single[0] = 1.000000f;

    _Static_assert(sizeof(array_func_arg_t) == 14 * sizeof(float),
        "Non-contiguous arg struct; please enable -fpack-struct or equivalent.");

    _Static_assert(sizeof(array_func_res_t) == 3 * sizeof(float),
        "Non-contiguous res struct; please enable -fpack-struct or equivalent.");

    return 0;
}

int array_func_step(array_func_arg_t* arg, array_func_res_t* res, array_func_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    // Marshal inputs to CasADi format
    const float* kernel_arg[array_func_SZ_ARG];
    kernel_arg[0] = &arg->zero_d;
    kernel_arg[1] = arg->one_d_single;
    kernel_arg[2] = arg->one_d_normal;
    kernel_arg[3] = arg->two_d_normal;
    kernel_arg[4] = (float*)&arg->edge_case;

    // Marshal outputs to CasADi format
    float* kernel_res[array_func_SZ_RES];
    kernel_res[0] = &res->sum;
    kernel_res[1] = &res->z;
    kernel_res[2] = (float*)&res->edge_out;

    // Call kernel function
    return array_func(kernel_arg, kernel_res, work->iw, work->w, 0);
}