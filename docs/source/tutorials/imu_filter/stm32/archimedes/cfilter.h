
#ifndef CFILTER_H
#define CFILTER_H

#include "cfilter_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float q[4];
    float rpy[3];
} attitude_t;

// Input arguments struct
typedef struct {
    attitude_t att;
    float gyro[3];
    float accel[3];
    float alpha;
    float dt;
} cfilter_arg_t;

// Output results struct
typedef struct {
    attitude_t att_fused;
} cfilter_res_t;

// Workspace struct
typedef struct {
    long int iw[cfilter_SZ_IW];
    float w[cfilter_SZ_W];
} cfilter_work_t;

// Runtime API
int cfilter_init(cfilter_arg_t* arg, cfilter_res_t* res, cfilter_work_t* work);
int cfilter_step(cfilter_arg_t* arg, cfilter_res_t* res, cfilter_work_t* work);


#ifdef __cplusplus
}
#endif

#endif // CFILTER_H