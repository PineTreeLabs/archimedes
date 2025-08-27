
#ifndef ARRAY_FUNC_H
#define ARRAY_FUNC_H

#include "array_func_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // empty was empty, no fields generated
    float single[1];
} edge_case_t;

// Input arguments struct
typedef struct {
    float zero_d;
    float one_d_single[1];
    float one_d_normal[5];
    float two_d_normal[6];
    edge_case_t edge_case;
} array_func_arg_t;

// Output results struct
typedef struct {
    float sum;
    float z;
    edge_case_t edge_out;
} array_func_res_t;

// Workspace struct
typedef struct {
    long int iw[array_func_SZ_IW];
    float w[array_func_SZ_W];
} array_func_work_t;

// Runtime API
int array_func_init(array_func_arg_t* arg, array_func_res_t* res, array_func_work_t* work);
int array_func_step(array_func_arg_t* arg, array_func_res_t* res, array_func_work_t* work);


#ifdef __cplusplus
}
#endif

#endif // ARRAY_FUNC_H