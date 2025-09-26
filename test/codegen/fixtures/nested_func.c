
#include <string.h>
#include "nested_func.h"

int nested_func_init(nested_func_arg_t* arg, nested_func_res_t* res, nested_func_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    /* Initialize inputs */
    memset(arg, 0, sizeof(nested_func_arg_t));

    /* Initialize outputs */
    memset(res, 0, sizeof(nested_func_res_t));

    /* Nonzero assignments */
    arg->scalar = 42.000000f;
    arg->arr[0] = 1.000000f;
    arg->arr[1] = 2.000000f;
    arg->arr[2] = 3.000000f;
    arg->clusters[0].center.x = 1.000000f;
    arg->clusters[0].center.y = 2.000000f;
    arg->clusters[0].points[0].x = 1.000000f;
    arg->clusters[0].points[0].y = 2.000000f;
    arg->clusters[0].points[1].x = 2.000000f;
    arg->clusters[0].points[1].y = 3.000000f;
    arg->clusters[0].points[2].x = 3.000000f;
    arg->clusters[0].points[2].y = 4.000000f;
    arg->clusters[0].weights[0] = 0.100000f;
    arg->clusters[0].weights[1] = 0.200000f;
    arg->clusters[0].weights[2] = 0.300000f;
    arg->clusters[1].center.x = 4.000000f;
    arg->clusters[1].center.y = 5.000000f;
    arg->clusters[1].points[0].x = 4.000000f;
    arg->clusters[1].points[0].y = 5.000000f;
    arg->clusters[1].points[1].x = 5.000000f;
    arg->clusters[1].points[1].y = 6.000000f;
    arg->clusters[1].points[2].x = 6.000000f;
    arg->clusters[1].points[2].y = 7.000000f;
    arg->clusters[1].weights[0] = 0.400000f;
    arg->clusters[1].weights[1] = 0.500000f;
    arg->clusters[1].weights[2] = 0.600000f;

    _Static_assert(sizeof(nested_func_arg_t) == 26 * sizeof(float),
        "Non-contiguous arg struct; please enable -fpack-struct or equivalent.");

    _Static_assert(sizeof(nested_func_res_t) == 1 * sizeof(float),
        "Non-contiguous res struct; please enable -fpack-struct or equivalent.");

    return 0;
}

int nested_func_step(nested_func_arg_t* arg, nested_func_res_t* res, nested_func_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    // Marshal inputs to CasADi format
    const float* kernel_arg[nested_func_SZ_ARG];
    kernel_arg[0] = &arg->scalar;
    kernel_arg[1] = arg->arr;
    kernel_arg[2] = (float*)arg->clusters;

    // Marshal outputs to CasADi format
    float* kernel_res[nested_func_SZ_RES];
    kernel_res[0] = &res->z;

    // Call kernel function
    return nested_func(kernel_arg, kernel_res, work->iw, work->w, 0);
}