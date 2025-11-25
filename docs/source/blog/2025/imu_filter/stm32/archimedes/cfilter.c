
#include <string.h>
#include "cfilter.h"

int cfilter_init(cfilter_arg_t* arg, cfilter_res_t* res, cfilter_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    /* Initialize inputs */
    memset(arg, 0, sizeof(cfilter_arg_t));

    /* Initialize outputs */
    memset(res, 0, sizeof(cfilter_res_t));

    /* Nonzero assignments */
    arg->att.q[0] = 1.000000f;
    arg->gyro[0] = 0.010000f;
    arg->gyro[1] = 0.020000f;
    arg->gyro[2] = 0.030000f;
    arg->accel[2] = -1.000000f;
    arg->alpha = 0.980000f;
    arg->dt = 0.010000f;

    _Static_assert(sizeof(cfilter_arg_t) == 15 * sizeof(float),
        "Non-contiguous arg struct; please enable -fpack-struct or equivalent.");

    _Static_assert(sizeof(cfilter_res_t) == 7 * sizeof(float),
        "Non-contiguous res struct; please enable -fpack-struct or equivalent.");

    return 0;
}

int cfilter_step(cfilter_arg_t* arg, cfilter_res_t* res, cfilter_work_t* work) {
    if (!arg || !res || !work) {
        return -1; // Invalid pointers
    }

    // Marshal inputs to CasADi format
    const float* kernel_arg[cfilter_SZ_ARG];
    kernel_arg[0] = (float*)&arg->att;
    kernel_arg[1] = arg->gyro;
    kernel_arg[2] = arg->accel;
    kernel_arg[3] = &arg->alpha;
    kernel_arg[4] = &arg->dt;

    // Marshal outputs to CasADi format
    float* kernel_res[cfilter_SZ_RES];
    kernel_res[0] = (float*)&res->att_fused;

    // Call kernel function
    return cfilter(kernel_arg, kernel_res, work->iw, work->w, 0);
}