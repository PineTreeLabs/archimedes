#ifndef FUNC_H
#define FUNC_H

#include "func_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

// Input arguments struct
typedef struct {
    float x[2];
    float y;
} func_arg_t;

// Output results struct
typedef struct {
    float x_new[2];
    float z[2];
} func_res_t;

// Workspace struct
typedef struct {
    long int iw[func_SZ_IW];
    float w[func_SZ_W];
} func_work_t;

// Runtime API
int func_init(func_arg_t* arg, func_res_t* res, func_work_t* work);
int func_step(func_arg_t* arg, func_res_t* res, func_work_t* work);


#ifdef __cplusplus
}
#endif

#endif // FUNC_H